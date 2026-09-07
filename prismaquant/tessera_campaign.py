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

The W4A4 leg is the **served static contract**, not the registry's dynamic
RTN.  ``tessera.serving.nvfp4_route`` reads the artifact's
``trellis_input_global_scale`` and calls vLLM's compiled ``scaled_fp4_quant``
-- a static-global-scale operation whose per-16 block scales are stored as
UE4M3 -- so an E2M1 rung here is scored through
``format_registry.nvfp4_activation_qdq_served`` at the unit's own calibrated
``input_global_scale`` (fused-sibling unified, the same joint the exporter's
scale file carries).  NVFP4's registry callback is a dynamic FP32-scale RTN
that never snaps through UE4M3 and rounds midpoints differently; pricing with
it prices an activation tensor the runtime does not execute
(RobTand/prismaquant#194).  A missing scale refuses rather than falling back
to the dynamic quantiser.

Rendering identity, and the wire
--------------------------------
The render is not ``render_tessera_weight``'s reconstruction; it is
``read_unit_artifact(encode_linear(...).blob)`` (or the equivalent batch
entry) -- **the bytes, decoded**.  So the
cache entry holds the wire beside the dequantised render. Checkpoint resume
verifies those bytes against their producer input receipt. That proves the
cached wire is the priced wire. The packed producer-plan/cached-wire bridge
carries those receipts through allocation and export; actual export/serve
qualification remains the measurement tracked by PrismaQuant #183.

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lane_eligibility import ServingContext

from . import tessera_hessian as th
from .nvfp4_activation_contract import (
    ActivationScaleContractError as _OwnedActivationScaleContractError,
)
from .tessera_expert_projection import EXPERT_WIRES_KEY, POPULATION_KEY, PROJECTION_KEY

__all__ = [
    "CENSUS_SCHEMA",
    "EXIT_EMPTY_MENU",
    "SCHEMA",
    "UNITS_SCHEMA",
    "UNITS_SCHEMA_V2",
    "CampaignAnchor",
    "ExpertPopulation",
    "anchor_group_key",
    "anchor_schedule",
    "audit_subsample",
    "calibration_census",
    "campaign_cost_payload",
    "campaign_population_block",
    "census_max_abs",
    "census_token_counts",
    "contract_source_label",
    "draw_stack_sample",
    "load_calibration_census",
    "load_unit_selection",
    "selection_stack_samples",
    "main",
    "next_anchor_rate",
    "report_empty_menus",
    "require_census_draw",
    "resolve_anchor_groups",
    "round_one_rates",
    "select_anchor_groups",
    "write_export_inputs",
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
    #: Encoding wall time; joined batches apportion it equally among units.
    seconds: float
    #: Was a Hessian actually applied to these bytes? Admission comes from the
    #: producer's ActivationSource settings for this rung's scale plane, via
    #: rung_accepts_hessian, not a campaign-owned plane roster. Stamped per
    #: measured row so weights-only and H-aware results cannot be conflated.
    hessian_applied: bool = False
    #: The static NVFP4 ``input_global_scale`` this anchor's A side was scored
    #: under, when the rung's route executes the static UE4M3 contract; None
    #: on every other route.  Carried per row so the price's activation
    #: identity survives into the cost table beside the Hessian identity.
    input_global_scale: "float | None" = None
    encoding_batch_size: int = 1


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


def parse_rate_band(text) -> "tuple[int, int] | None":
    """``"lo,hi"`` in q256 body-rate units, or None when unset."""
    if text is None or str(text).strip() == "":
        return None
    parts = str(text).split(",")
    if len(parts) != 2:
        raise RuntimeError(f"--rate-band {text!r}: want 'lo,hi' in q256 units")
    try:
        lo, hi = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise RuntimeError(f"--rate-band {text!r}: not two integers") from exc
    if lo <= 0 or hi <= 0 or lo > hi:
        raise RuntimeError(f"--rate-band {text!r}: want 0 < lo <= hi")
    return lo, hi


def round_one_rates(allowed: "Sequence[int]", *, band, anchors: int,
                    snap) -> list[int]:
    """Where round one puts this family's anchors on this group's grid.

    Without a band the schedule spans the family's whole realisable range,
    which is what every artifact before 2026-09-06 was priced under.  With
    one, the anchors go at the ends of the **band**, and the reason is
    measured rather than economised: the surface is not one line over the
    whole 1--8 bit range.  On campaign-01 a two-anchor fit that spans the full
    range and drops the interior anchor of a 1 / 4.5 / 8-bit triple misses the
    interior by median 0.32 and p90 0.76 log2, with the same sign at both
    ends -- curvature, not noise -- while the same construction inside a
    one-bit bracket lands at median 0.08.  So the fix is not more anchors over
    a range nothing will be allocated in; it is the same two anchors placed
    around the rates the artifact will actually use.

    A family whose realisable rungs do not reach the band gets no anchor here
    and is left to say so, rather than being priced at a rate outside it.
    """
    if not allowed:
        return []
    if band is None:
        return sorted({r for r in (snap(rate, allowed) for rate in
                                   anchor_schedule(allowed[0], allowed[-1], anchors))
                       if r is not None})
    lo, hi = band
    inside = [rate for rate in allowed if lo <= int(rate) <= hi]
    if not inside:
        return []
    return sorted({snap(inside[0], allowed), snap(inside[-1], allowed)} - {None})


def audit_extra_rate(allowed: "Sequence[int]", placed: "Sequence[int]",
                     *, snap) -> "int | None":
    """The one interior rung an audit unit measures on top of ``placed``.

    Two anchors define a line and can never disagree with it, so a bracket
    priced at its ends carries no evidence about its own interpolation.  The
    audit subsample buys that evidence for the price of one extra encode per
    ten sampled units: a third anchor in the middle of the bracket, which the
    existing leave-one-anchor-out report then scores against the line the
    other two draw.  Returns None when the bracket has no interior rung to
    put it on.
    """
    if len(placed) < 2:
        return None
    lo, hi = int(min(placed)), int(max(placed))
    interior = [int(r) for r in allowed if lo < int(r) < hi]
    if not interior:
        return None
    return snap((lo + hi) // 2, interior)


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

class ActivationScaleContractError(_OwnedActivationScaleContractError):
    """A static-activation-contract rung has no calibrated input scale.

    Its own class for the same reason ``HessianContractError`` has one: the
    anchor loop absorbs per-anchor failures with ``except Exception:
    continue``, and a scale-contract refusal is about every W4A4 row this run
    would write, not about one anchor.  Falling back to the registry's dynamic
    FP32-scale quantiser instead would price an activation regime the serve
    does not execute -- exactly the defect this refusal exists to make loud
    (RobTand/prismaquant#194).

    The same refusal, raised by the assignment-KL hooks and the production
    cache scorer, is the owner's
    ``nvfp4_activation_contract.ActivationScaleContractError``; this is that
    class under the campaign's historical name (#205), so a caller catching
    either sees one contract error.
    """


def _encode_and_render(weight, format_name: str, *, activation_kwargs=None,
                       hessian_required: bool = True, recipe=None):
    """``(render, blob)`` for one rung: the bytes, and what they decode to.

    A thin adapter over ``tessera_render.encode_tessera_unit``, which is the
    shared scalar/batch adapter that calls Tessera's byte path.  Everything
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
    static_input_scale: "float | None" = None,
):
    """Render one rung, price it as served, and store the wire beside it.

    ``static_input_scale`` is the unit's calibrated NVFP4
    ``input_global_scale`` (fused-sibling unified), required whenever the
    rung's route executes the static UE4M3 activation contract and ignored on
    every other route.  The refusal for a missing one runs BEFORE the encode,
    because the encode is the expensive half and the refusal is about the
    whole run.

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
    prepared = _prepare_anchor(
        qname=qname, format_name=format_name,
        activation_kwargs_for=activation_kwargs_for,
        hessian_required=hessian_required, static_input_scale=static_input_scale)
    started = time.time()
    render, blob = _encode_and_render(
        weight, format_name, recipe=prepared["wire"],
        activation_kwargs=prepared["activation_kwargs"],
        hessian_required=prepared["hessian_required"])
    return _finish_anchor(qname=qname, weight=weight, activations=activations,
        format_name=format_name, cache=cache, wire_dir=wire_dir,
        prepared=prepared, render=render, blob=blob, elapsed=time.time() - started)


def _prepare_anchor(*, qname, format_name, activation_kwargs_for,
                    hessian_required, static_input_scale):
    """Admit each unit's Hessian and served activation contract before encode."""
    from . import format_registry as fr
    from .tessera_formats import (
        parse_tessera_format_name, tessera_serving_route, tessera_wire_recipe,
    )
    from .tessera_render import HessianContractError

    from .tessera_render import rung_accepts_hessian

    spec = fr.get_format(format_name)
    family, rung = parse_tessera_format_name(format_name)
    # ONE resolve for this anchor. The plane the kwargs are built on, the plane
    # the predicate reads and the plane the encode writes are this object --
    # not three lookups that agree only while nothing clears the recipe memo.
    wire = tessera_wire_recipe(family, rung)
    # The A side, as served.  A route with a STATIC activation contract
    # executes vLLM's static-global-scale ``scaled_fp4_quant`` (UE4M3 block
    # scales) against the artifact's ``trellis_input_global_scale``; every
    # other route keeps the serving format's own dynamic quantiser, taken by
    # reference from the spec.
    #
    # WHICH of the two is the SPEC's answer, never a compare of the route's
    # source-format name against ``"NVFP4"`` (#205's rule, #221's fix): the
    # spec resolved above already carries the contract, because
    # ``synthesize_tessera_spec`` derived it from the registry row the route
    # names.  A Tessera rung routed through the same kernel has the same
    # contract and a different name, and the day a second registry row gets a
    # contract, a name compare here would price it under a quantiser the
    # runtime never runs while the cache scorer and the KL hooks -- which
    # already read the row -- refuse the same unit.
    route = tessera_serving_route(family, wire, rung)
    contract = spec.static_activation_contract
    if contract is not None:
        if static_input_scale is None or not float(static_input_scale) > 0:
            raise ActivationScaleContractError(
                f"{qname} {format_name}: this rung's route executes the "
                f"static activation contract {contract.execution} "
                f"({route.contract}) and no calibrated input_global_scale was "
                "supplied. Scoring it with the registry's dynamic FP32-scale "
                "quantiser would price an activation regime the serve does "
                "not execute; a lookup that misses must refuse, not fall back."
            )
        input_scale = float(static_input_scale)
        # The contract's own oracle, not a second spelling of it.
        activation_qdq = (
            lambda x: contract.quantize_dequantize(x, input_scale))
    else:
        input_scale = None
        activation_qdq = spec.activation_quantize_dequantize
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
    return dict(spec=spec, family=family, rung=rung, wire=wire,
                activation_qdq=activation_qdq, input_scale=input_scale,
                activation_kwargs=activation_kwargs,
                hessian_required=hessian_required)


def _finish_anchor(*, qname, weight, activations, format_name, cache, wire_dir,
                   prepared, render, blob, elapsed, encoding_batch_size=1):
    """Score decoded bytes and publish the existing cache/wire entries."""
    import torch
    from .production_weight_cache import (
        _local_forward_render_score, _store_rendered_weight_entry)

    spec, family, rung = (prepared[key] for key in ("spec", "family", "rung"))
    activation_qdq, input_scale = (prepared[key] for key in (
        "activation_qdq", "input_scale"))
    score, metric, quantized, _clipped = _local_forward_render_score(
        reference_weight=weight,
        rendered_weight=render,
        activations=activations,
        activation_quantize=activation_qdq,
        # No pre-clip on top: the static route's clamp lives inside the served
        # oracle (``nvfp4_activation_qdq_served`` clamps at the stored UE4M3
        # scale), and the dynamic routes have no calibrated clip at all.
        activation_max_abs=None,
    )
    if metric != "output_mse":
        raise RuntimeError(f"unexpected render-score metric {metric!r}")
    hessian_applied = bool(prepared["activation_kwargs"])

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
    wire_path = _wire_path(wire_dir, qname, format_name)
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
        input_global_scale=input_scale,
        encoding_batch_size=encoding_batch_size,
    )


def _measure_anchor_batch(*, qnames, weights, activations, format_name,
                          cache, wire_dir, activation_kwargs_for=None,
                          hessian_required=True, static_input_scales=None):
    """One producer batch, with the scalar path's per-unit gates and storage."""
    from .tessera_render import encode_tessera_units

    if not qnames or len(set(qnames)) != len(qnames):
        raise ValueError("anchor batch needs distinct nonempty unit names")
    if len(weights) != len(qnames) or len(activations) != len(qnames):
        raise ValueError("anchor batch needs one weight and activation input per unit")
    prepared = [_prepare_anchor(
        qname=name, format_name=format_name,
        activation_kwargs_for=activation_kwargs_for,
        hessian_required=hessian_required,
        static_input_scale=(static_input_scales or {}).get(name)) for name in qnames]
    started = time.time()
    encoded = encode_tessera_units(
        weights, format_name, recipe=prepared[0]["wire"],
        activation_kwargs=[entry["activation_kwargs"] for entry in prepared],
        hessian_required=prepared[0]["hessian_required"])
    # Batch wall time is apportioned, not attributed as a measured per-unit
    # duration. Summing anchor.seconds still reports the actual encoding time.
    elapsed = (time.time() - started) / len(qnames)
    return [_finish_anchor(qname=name, weight=weight, activations=acts,
        format_name=format_name, cache=cache, wire_dir=wire_dir, prepared=entry,
        render=render, blob=blob, elapsed=elapsed, encoding_batch_size=len(qnames))
        for name, weight, acts, entry, (render, blob) in zip(
            qnames, weights, activations, prepared, encoded)]


def _anchor_batches(pending, *, weights, expert_members, batch_size):
    """Bound compatible expert encodes inside this action; never assign hosts."""
    if batch_size < 1:
        raise ValueError("anchor batch size must be positive")
    if batch_size == 1:
        return [[item] for item in pending]
    groups = {}
    for item in pending:
        name, family, rung = item
        weight = weights[name]
        key = ((family, rung, tuple(weight.shape), weight.dtype, weight.device)
               if name in expert_members else (name, family, rung))
        groups.setdefault(key, []).append(item)
    return [group[start:start + batch_size] for group in groups.values()
            for start in range(0, len(group), batch_size)]


# ---------------------------------------------------------------------------
# The routed-expert stack, and the sample that prices it
# ---------------------------------------------------------------------------
#
# A routed MoE stack is ONE decision: vLLM loads a packed ``[E, M, N]`` expert
# tensor under a single quantization scheme, and PrismaQuant's union-find
# serving-unit promotion already enforces that (``allocator_solver.
# _packed_groups_by_profile`` keys ``...experts.gate_up_proj`` and
# ``...experts.down_proj`` into the same ``__packed_format__`` group).  The
# campaign measures separate expert Linears, optionally encoding compatible
# units together. A full census still requires every expert's wire and score.
#
# So the campaign may measure a SAMPLE of experts and this module estimates the
# stack from it.  Two things have to be right for that to be honest:
#
# 1. The estimator must be unbiased for the stack's total, not for "the average
#    expert".  With a probability-proportional-to-size draw the Horvitz-Thompson
#    estimator is the one that is: ``T_hat = sum_{e in S} y_e / pi_e``.
#
# 2. The number written into the row must be the quantity the ALLOCATOR will
#    multiply correctly.  ``allocator_solver.predicted_dloss`` prices a row as
#    ``0.5 * h_trace * output_mse``, and it reads ``h_trace`` from the PROBE, so
#    the stack row is multiplied by the packed row's own ``h_trace``.  The
#    quantity that makes that product reproduce the stack's summed per-expert
#    dloss is therefore
#
#        output_mse_stack = ( sum_e h_e * mse_e ) / h_stack
#
#    i.e. the h-weighted MEAN expert MSE, not the sum.  Two independent facts
#    in this repo fix that convention rather than taste:
#
#      * the probe's own identity ``sum_e h_trace_per_expert[e] == h_trace``,
#        which holds to float32 storage precision on all 44 packed rows of
#        ``pq275-2026-09-06/probe-02/probe.pkl`` (2.0e-7 relative on layer 18),
#        so the weights sum to the multiplier the allocator will apply; and
#      * ``allocator_candidates``' own super-item aggregation, which inverts the
#        same product the same way when it has to express a group's summed
#        predicted dloss as one member-shaped MSE:
#        ``effective_mse = base_pred / (0.5 * sum_h)``.
#
# A per-expert member's Fisher weight is ``h_trace_per_expert[e] / R``, where R
# is the number of projections the packed parameter splits into -- the SAME
# split ``tier2_per_expert_counterfactual.expand_packed_expert_rows`` uses, and
# R comes from the profile's ``packed_expert_projection_names``, never from a
# hardcoded ``gate_up_proj -> 2``.


class StackSampleError(ValueError):
    """A routed-stack sampling record cannot be turned into an honest row."""


@dataclass(frozen=True)
class StackExpertSample:
    """Which experts of one packed stack were measured, and with what weight.

    Built by the campaign driver from the PROBE row plus its own draw --
    ``stack_sample_from_probe`` is the constructor that reads the profile so no
    caller has to spell a projection split.  Everything here that describes the
    stack (``num_experts``, ``stack_h_trace``, ``h_trace_per_expert``,
    ``packed_experts_module``) is copied from the probe row, so the row this
    record prices and the row the allocator multiplies are the same row.
    """

    #: The packed parameter's qname -- the key the stack row is written under,
    #: and a key the probe already has (e.g.
    #: ``model.layers.18.feed_forward.experts.gate_up_proj``).
    packed_qname: str
    #: ``_packed_experts_module`` from the probe row, copied onto the cost row
    #: so ``tessera_serving_scope.unit_structure_from_stats`` resolves
    #: ``routed_moe`` on its EXISTING packed branch rather than being taught a
    #: new topology (principle 14: the scope may not invent structure).
    packed_experts_module: str
    #: ``gate_up_proj`` / ``down_proj``: the packed parameter, used only to ask
    #: the profile for its projection names.
    packed_param: str
    num_experts: int
    #: The probe row's ``h_trace`` -- the exact multiplier the allocator applies.
    stack_h_trace: float
    #: The probe row's ``h_trace_per_expert``; must sum to ``stack_h_trace``.
    h_trace_per_expert: tuple[float, ...]
    #: Expert ids actually encoded, ascending.
    sampled_experts: tuple[int, ...]
    #: pi_e for every sampled expert. 1.0 marks a certainty stratum (or a
    #: census), which contributes zero sampling variance.
    inclusion_prob: "Mapping[int, float]"
    #: expert id -> the anchor qnames measured for it, one per projection.
    members: "Mapping[int, tuple[str, ...]]"
    #: The draw's seed, and its design name, carried for reproduction.
    seed: int
    design: str = "pps_wor"

    @property
    def is_census(self) -> bool:
        return len(self.sampled_experts) == self.num_experts and all(
            float(self.inclusion_prob[e]) == 1.0 for e in self.sampled_experts)


def stack_sample_from_probe(
    packed_qname: str,
    probe_row: "Mapping[str, object]",
    profile,
    *,
    sampled_experts,
    inclusion_prob,
    seed: int,
    design: str = "pps_wor",
) -> StackExpertSample:
    """Build a sampling record from the PROBE row and the model profile.

    The caller supplies only its draw.  Everything structural -- the expert
    count, the Fisher weights, the packed module, and the per-expert member
    qnames -- is read here from the probe row and the profile's packed-expert
    accessors, because a driver-supplied weight that drifts from the probe
    would break the currency invisibly: the allocator would still multiply by
    the probe's ``h_trace``.
    """
    packed_param = str(probe_row.get("_packed_param") or "")
    module = probe_row.get("_packed_experts_module")
    num_experts = int(probe_row.get("num_experts", 0) or 0)
    stack_h = float(probe_row.get("h_trace", 0.0) or 0.0)
    per_expert = probe_row.get("h_trace_per_expert")
    if not isinstance(module, str) or not module:
        raise StackSampleError(
            f"{packed_qname}: probe row has no _packed_experts_module; this is "
            "not a packed routed stack and must not be priced as one")
    if num_experts <= 0:
        raise StackSampleError(f"{packed_qname}: probe row has no num_experts")
    if not isinstance(per_expert, Sequence) or len(per_expert) != num_experts:
        raise StackSampleError(
            f"{packed_qname}: stack pricing requires "
            f"h_trace_per_expert[{num_experts}] from the probe")
    if not packed_param:
        raise StackSampleError(f"{packed_qname}: probe row has no _packed_param")
    if not packed_qname.endswith(packed_param):
        raise StackSampleError(
            f"{packed_qname}: does not end in its own _packed_param "
            f"{packed_param!r}; refusing to guess the stem")
    # The projection split comes from the profile, so a family whose leaves are
    # ``w1/w3/w2`` and one whose leaves are ``gate_proj/up_proj/down_proj`` are
    # both handled without this module knowing either spelling.
    roles = tuple(profile.packed_expert_projection_names(packed_param))
    if not roles:
        raise StackSampleError(
            f"{packed_qname}: profile declares no projections for "
            f"{packed_param!r}")
    stem = packed_qname[: -len(packed_param)].rstrip(".")
    ordered = tuple(sorted(int(e) for e in sampled_experts))
    if len(set(ordered)) != len(ordered):
        raise StackSampleError(f"{packed_qname}: duplicate sampled expert id")
    members = {e: tuple(f"{stem}.{e}.{role}" for role in roles) for e in ordered}
    return StackExpertSample(
        packed_qname=str(packed_qname),
        packed_experts_module=module,
        packed_param=packed_param,
        num_experts=num_experts,
        stack_h_trace=stack_h,
        h_trace_per_expert=tuple(float(v) for v in per_expert),
        sampled_experts=ordered,
        inclusion_prob={int(e): float(p) for e, p in dict(inclusion_prob).items()},
        members=members,
        seed=int(seed),
        design=str(design),
    )


#: ``float32`` machine epsilon.  The probe accumulates the packed row's
#: ``h_trace`` and its ``h_trace_per_expert`` vector from the same raw sums and
#: divides both by the same global token count
#: (``sensitivity_probe.finalize_fisher_stats``), so the two agree exactly in
#: exact arithmetic.  They are stored at ``float32``, which is the only reason
#: they differ at all, so the admissible discrepancy is a property of that
#: dtype and of how many terms are summed -- not a tolerance anyone chose.
#: Measured on ``probe-02``'s two layer-18 packed rows: 2.0e-7 and 6.6e-8
#: relative, against the bound below of 3.8e-6 at E=32.
_FLOAT32_EPS = 1.1920928955078125e-07


def _validate_stack_sample(sample: StackExpertSample) -> None:
    """Refuse a record whose weights or probabilities cannot carry an estimate."""
    q = sample.packed_qname
    if len(sample.h_trace_per_expert) != sample.num_experts:
        raise StackSampleError(
            f"{q}: h_trace_per_expert has {len(sample.h_trace_per_expert)} "
            f"entries for {sample.num_experts} experts")
    if any(not math.isfinite(h) or h < 0.0 for h in sample.h_trace_per_expert):
        raise StackSampleError(
            f"{q}: per-expert Fisher weights must be finite and nonnegative")
    total = math.fsum(sample.h_trace_per_expert)
    if not math.isfinite(sample.stack_h_trace) or sample.stack_h_trace <= 0.0:
        raise StackSampleError(f"{q}: probe h_trace must be finite and positive")
    # E terms summed at float32 storage precision: the worst-case accumulated
    # relative error is E * eps, so that IS the bound, computed per stack.
    tolerance = sample.num_experts * _FLOAT32_EPS * sample.stack_h_trace
    if abs(total - sample.stack_h_trace) > tolerance:
        # The whole currency rests on this identity; a row written against a
        # broken one would be multiplied by a number its weights do not sum to.
        raise StackSampleError(
            f"{q}: sum(h_trace_per_expert)={total!r} does not equal the probe "
            f"row's h_trace={sample.stack_h_trace!r}; the stack row's currency "
            "assumes the per-expert weights sum to the multiplier the "
            f"allocator applies (tolerance {tolerance!r} = "
            f"{sample.num_experts} * float32 eps)")
    if not sample.sampled_experts:
        raise StackSampleError(f"{q}: no sampled experts")
    if len(set(sample.sampled_experts)) != len(sample.sampled_experts):
        raise StackSampleError(f"{q}: duplicate sampled expert id")
    for e in sample.sampled_experts:
        if not 0 <= e < sample.num_experts:
            raise StackSampleError(f"{q}: expert id {e} out of range")
        if e not in sample.inclusion_prob:
            raise StackSampleError(f"{q}: expert {e} has no inclusion probability")
    # Older callers carry only sampled probabilities. When the full frame is
    # supplied, validate the unsampled entries too: zero-probability positive
    # contributions and omitted certainty units both bias the stack total.
    for e, probability in sample.inclusion_prob.items():
        if not 0 <= e < sample.num_experts:
            raise StackSampleError(f"{q}: expert id {e} out of range")
        pi = float(probability)
        if pi == 0.0:
            # A zero-probability unit contributes an exactly-zero term ONLY if
            # its own weight is zero (a never-routed expert). Otherwise it is a
            # unit the design can never draw, and HT is biased by exactly its
            # contribution -- silently.
            if sample.h_trace_per_expert[e] != 0.0:
                raise StackSampleError(
                    f"{q}: expert {e} has inclusion probability 0 but Fisher "
                    f"weight {sample.h_trace_per_expert[e]!r}; a unit that "
                    "cannot be drawn biases the estimate")
            continue
        if not 0.0 < pi <= 1.0:
            raise StackSampleError(
                f"{q}: expert {e} inclusion probability {pi!r} outside (0, 1]")
        if pi == 1.0 and e not in sample.sampled_experts:
            raise StackSampleError(f"{q}: certainty expert {e} is absent from the sample")


def _stack_member_weight(sample: StackExpertSample, expert: int, roles: int) -> float:
    """The Fisher weight of ONE measured member of one expert.

    ``h_trace_per_expert[e] / R`` -- the same equal split
    ``expand_packed_expert_rows`` applies, so a census through this path and an
    expansion through that one price the stack identically.
    """
    return float(sample.h_trace_per_expert[expert]) / float(roles)


def _horvitz_thompson_stack(
    sample: StackExpertSample,
    member_dloss: "Mapping[int, float]",
) -> "tuple[float, float, int]":
    """Estimate ``sum_e h_e * mse_e`` and its standard error from the sample.

    ``member_dloss`` maps a sampled expert id to that expert's h-weighted
    contribution ``y_e = sum_roles (h_e/R) * mse_role`` already summed over the
    packed parameter's projections.

    Returns ``(T_hat, stderr, m)`` where ``m`` is the size of the random
    stratum.

    The point estimate is Horvitz-Thompson, ``sum_{e in S} y_e / pi_e``, which
    is unbiased for the stack total under any design with known positive
    inclusion probabilities.

    The VARIANCE is Hartley-Rao's, over the non-certainty stratum only:

        v = m/(m-1) * sum_{e in S_R} (1 - (m-1)/m * pi_e)
                                     * (y_e/pi_e - T_R/m)^2

    A unit with ``pi_e == 1`` is in every possible sample, so it contributes
    exactly zero sampling variance and is excluded from the sum. This is a
    plug-in Hartley-Rao approximation for randomized-order systematic PPS,
    ``pi_i = min(1, c*h_i)``, after removing the take-all stratum. It is not
    an exact design-unbiased variance estimator. Exact Sen-Yates-Grundy would
    require joint inclusion probabilities averaged over the randomized order;
    zero probabilities conditional on one fixed order do not establish zeros
    under the randomized design.  The Hansen-Hurwitz with-replacement form
    is NOT used: it ignores both the finite-population correction and the
    certainty stratum, and overstates the standard error by 25-48% at E=32
    (simulated on the LFM2.5 layer-18 Fisher vector). The approximation was
    conservative in those simulations (+1% to +15%); that is not a guarantee
    for other populations. With equal weights, its expected variance is
    (N-m+1)/(N-m) times the exact SRS variance, reaching twice the exact value
    for m=N-1. The current allocator does not consume this uncertainty field.
    """
    certainty = [e for e in sample.sampled_experts
                 if float(sample.inclusion_prob[e]) == 1.0]
    random = [e for e in sample.sampled_experts
              if 0.0 < float(sample.inclusion_prob[e]) < 1.0]
    total = math.fsum(member_dloss[e] for e in certainty)
    t_random = math.fsum(member_dloss[e] / float(sample.inclusion_prob[e])
                         for e in random)
    total += t_random
    m = len(random)
    if m == 0:
        # Every measured unit was certain: a census, or a fully take-all
        # stratum. The estimate IS the total and the error is a true zero.
        return total, 0.0, 0
    if m < 2:
        # 0.0 already means "no sampling error" on this row's siblings, and a
        # single random draw is not that. The draw plan is supposed to refuse
        # m == 1 before anything is encoded; refusing again here keeps a plan
        # that did not from being laundered into a published zero.
        raise StackSampleError(
            f"{sample.packed_qname}: {m} non-certainty draw(s); a sampling "
            "variance needs at least two, and writing 0.0 would claim the "
            "zero that a census means")
    mean_r = t_random / m
    variance = (m / (m - 1)) * math.fsum(
        (1.0 - ((m - 1) / m) * float(sample.inclusion_prob[e]))
        * (member_dloss[e] / float(sample.inclusion_prob[e]) - mean_r) ** 2
        for e in random)
    return total, math.sqrt(max(variance, 0.0)), m


def _stack_menu(sample: StackExpertSample, menus: "Mapping[str, list]") -> list:
    """The menu the stack row is interpolated over.

    Preferred source is a menu the driver keyed at the packed qname.  Failing
    that the stack inherits its members' menu -- but only if every measured
    member offers the SAME rungs, because a stack is one decision and a menu
    that differs between the experts inside it is not a menu for that decision.
    """
    packed = menus.get(sample.packed_qname)
    if packed:
        return list(packed)
    seen: dict[tuple, list] = {}
    for expert in sample.sampled_experts:
        for member in sample.members[expert]:
            rungs = list(menus.get(member, []))
            key = tuple(sorted(
                (r.family, r.format_name, int(r.body_rate_q256)) for r in rungs))
            seen.setdefault(key, rungs)
    if not seen:
        return []
    if len(seen) > 1:
        raise StackSampleError(
            f"{sample.packed_qname}: the measured experts do not share one "
            f"menu ({len(seen)} distinct rung sets); a packed stack is a "
            "single serving decision and cannot be priced over a menu that "
            "differs between the experts inside it")
    return next(iter(seen.values()))


#: The anchor fields that must agree across every measured member of one stack
#: rung, mapped to the row field they are written to.  Taking member zero's
#: value the way the dense interpolation path takes ``ordered[0]``'s would let
#: one expert scored under a different activation contract, or without the
#: Hessian, disappear into a stack average.
_STACK_UNIFORM_FIELDS = (
    "family", "body_rate_q256", "activation_contract",
    "activation_quantized", "hessian_applied",
)


def _stack_cost_rows(
    sample: StackExpertSample,
    anchors: "Mapping[str, Mapping[str, list]]",
    menus: "Mapping[str, list]",
    hessian_identity: dict,
    refused: list,
    wire_backed: "frozenset[str] | set[str]" = frozenset(),
) -> "tuple[dict[str, dict], object]":
    """Build every measured + interpolated row for one packed stack."""
    from .tessera_rate_surface import (
        PROVENANCE_INTERPOLATED, PROVENANCE_MEASURED, TesseraRateSurface,
    )

    _validate_stack_sample(sample)
    q = sample.packed_qname
    roles = len(sample.members[sample.sampled_experts[0]])
    if roles <= 0:
        raise StackSampleError(f"{q}: sampled experts have no measured members")

    # (format_name) -> {expert -> [anchor, ...]}, one anchor per member.
    by_format: dict[str, dict[int, list]] = {}
    for expert in sample.sampled_experts:
        if len(sample.members[expert]) != roles:
            raise StackSampleError(
                f"{q}: expert {expert} contributes "
                f"{len(sample.members[expert])} members, expert "
                f"{sample.sampled_experts[0]} contributes {roles}")
        for member in sample.members[expert]:
            member_anchors = anchors.get(member)
            if not member_anchors:
                raise StackSampleError(
                    f"{q}: sampled member {member} has no measured anchors")
            for family_anchors in member_anchors.values():
                for anchor in family_anchors:
                    by_format.setdefault(
                        anchor.format_name, {}).setdefault(expert, []).append(anchor)

    rows: dict[str, dict] = {}
    measured_by_family: dict[str, list[tuple[int, float, float]]] = {}
    for format_name in sorted(by_format):
        per_expert = by_format[format_name]
        missing = [e for e in sample.sampled_experts
                   if len(per_expert.get(e, ())) != roles]
        if missing:
            # A rung measured on only some of the drawn experts is not a rung
            # this sample can price: HT needs every drawn unit's y_e.
            refused.append({
                "qname": q, "format_name": format_name,
                "reason": "stack_rung_incomplete_over_sample",
                "detail": (f"{len(missing)} of {len(sample.sampled_experts)} "
                           "sampled experts lack a full set of member anchors"),
                "missing_experts": missing,
            })
            continue
        contributing = [a for e in sample.sampled_experts for a in per_expert[e]]
        uniform = {}
        for field in _STACK_UNIFORM_FIELDS:
            values = {getattr(a, field) for a in contributing}
            if len(values) != 1:
                raise StackSampleError(
                    f"{q}/{format_name}: measured members disagree on {field} "
                    f"({sorted(map(repr, values))}); a stack row may not "
                    "average measurements taken under different contracts")
            uniform[field] = next(iter(values))
        member_dloss = {
            e: math.fsum(_stack_member_weight(sample, e, roles) * float(a.dloss)
                         for a in per_expert[e])
            for e in sample.sampled_experts
        }
        total, stderr, n_random = _horvitz_thompson_stack(sample, member_dloss)
        h_stack = float(sample.stack_h_trace)
        rows[format_name] = {
            # The h-weighted MEAN expert MSE -- see the section header. The
            # allocator multiplies this by the PROBE row's h_trace, which the
            # per-expert weights sum to, so 0.5*h_stack*output_mse reproduces
            # the stack's summed per-expert predicted dloss.
            "output_mse": total / h_stack,
            "output_mse_measured": True,
            "cost_source": "tessera_campaign_measured_stack_sample",
            "currency": CURRENCY,
            "tessera_provenance": PROVENANCE_MEASURED,
            "tessera_family": uniform["family"],
            "tessera_body_rate_q256": uniform["body_rate_q256"],
            "activation_quantized": uniform["activation_quantized"],
            "activation_contract": uniform["activation_contract"],
            # The Horvitz-Thompson standard error, in the SAME currency as
            # ``output_mse``.  REPORTED, NOT CONSUMED: the allocator's UCB
            # hedge (``allocator_candidates._super_item_ucb_hedge``) skips
            # rows priced from ``output_mse``, so no DP behaviour depends on
            # this field today.  It is written because a sampled price whose
            # sampling error is nowhere on the row is a sampled price nothing
            # can audit.
            "dloss_stderr": stderr / h_stack,
            "dloss_stderr_currency": CURRENCY,
            "dloss_stderr_consumed_by_allocator": False,
            # Copied from the PROBE row so the explicit Tessera serving scope
            # resolves ``routed_moe`` on ``unit_structure_from_stats``'s
            # existing packed branch. The scope is not taught a new topology;
            # it is handed the one the probe already recorded.
            "_packed_experts_module": sample.packed_experts_module,
            "num_experts": sample.num_experts,
            "encode_seconds": math.fsum(float(a.seconds) for a in contributing),
            "hessian_identity": {
                **hessian_identity, "applied": bool(uniform["hessian_applied"]),
            },
            "sampled_experts": {
                "design": sample.design,
                "seed": sample.seed,
                "n_experts": sample.num_experts,
                "n_sampled": len(sample.sampled_experts),
                "n_random_stratum": n_random,
                "experts": list(sample.sampled_experts),
                "inclusion_prob": {int(e): float(sample.inclusion_prob[e])
                                   for e in sample.sampled_experts},
                "packed_param": sample.packed_param,
                "projections_per_expert": roles,
                "estimator": "horvitz_thompson",
                "variance_estimator": (
                    "census" if n_random == 0 else "hartley_rao"),
                "h_trace_stack": h_stack,
                "h_trace_per_sampled_expert": {
                    int(e): float(sample.h_trace_per_expert[e])
                    for e in sample.sampled_experts},
                # The per-member evidence the stack row was estimated from,
                # kept on the row because the members are NOT cost keys: a
                # stack price whose members are unreadable is unauditable.
                "members": {
                    int(e): [
                        {"qname": a.qname, "dloss": float(a.dloss),
                         "wire_bytes": int(a.wire_bytes),
                         "memory_bytes": int(a.memory_bytes),
                         "input_global_scale": (
                             None if a.input_global_scale is None
                             else float(a.input_global_scale))}
                        for a in per_expert[e]]
                    for e in sample.sampled_experts},
            },
        }
        # No ``wire_bytes`` and no ``input_global_scale`` scalar on a stack
        # row, deliberately.  Both are per-expert facts: a sample has neither a
        # stack wire nor one A-side scale, and the per-member values are on the
        # row above.  ``tessera_menu.priced_static_scales`` will therefore find
        # no scale for a selected W4A4 stack and the export lane will refuse it
        # by name -- which is the correct refusal until the driver calibrates
        # a scale for every expert, sampled or not (reported, not papered over).
        measured_by_family.setdefault(uniform["family"], []).append(
            (int(uniform["body_rate_q256"]), total / h_stack, stderr / h_stack))

    # Interpolation, on the STACK's own anchors: the same monotone surface the
    # dense path uses, fitted to HT estimates rather than to one unit's
    # measurements, and refusing to extrapolate for the same reason.
    surfaces: dict[str, object] = {}
    measured_names = set(rows)
    for family, points in measured_by_family.items():
        ordered = sorted(points)
        try:
            surface = TesseraRateSurface(
                unit_name=q, family=family, layout="tight", currency=CURRENCY,
                anchor_q256=tuple(p[0] for p in ordered),
                anchor_dloss=tuple(p[1] for p in ordered),
                anchor_stderr=tuple(p[2] for p in ordered),
            )
        except Exception as exc:
            refused.append({
                "qname": q, "family": family,
                "reason": "non_interpolable_anchors",
                "detail": str(exc),
                "anchor_q256": [p[0] for p in ordered],
                "anchor_dloss": [p[1] for p in ordered],
            })
            continue
        surfaces[family] = surface
        if sample.is_census and all(
                member in wire_backed for members in sample.members.values()
                for member in members):
            # A full census handed to the cached-wire exporter can select
            # only rungs with actual member bytes. Keep its surface for
            # diagnostics, as the dense path does, but never price an
            # interpolated rung as though the corresponding wires existed.
            continue
        low, high = surface.q256_range
        template = next(r for r in rows.values()
                        if r["tessera_family"] == family)
        for rung in _stack_menu(sample, menus):
            if rung.family != family or rung.format_name in measured_names:
                continue
            if not low <= rung.body_rate_q256 <= high:
                continue
            rows[rung.format_name] = {
                "output_mse": surface.predict(rung.body_rate_q256),
                "output_mse_measured": False,
                "cost_source": "tessera_campaign_interpolated",
                "currency": CURRENCY,
                "tessera_provenance": PROVENANCE_INTERPOLATED,
                "tessera_family": family,
                "tessera_body_rate_q256": rung.body_rate_q256,
                "activation_contract": rung.admission.activation_contract,
                "_packed_experts_module": sample.packed_experts_module,
                "num_experts": sample.num_experts,
                "hessian_identity": {
                    **hessian_identity,
                    "applied": bool(template["hessian_identity"]["applied"]),
                },
                # An interpolated stack row inherits the sample its bracketing
                # anchors were estimated from; it is a prediction ABOUT that
                # sample, so the sample travels with it.
                "sampled_experts": {
                    **{k: v for k, v in template["sampled_experts"].items()
                       if k != "members"},
                    "interpolated_from": sorted(measured_names),
                },
            }
    return rows, surfaces


# ---------------------------------------------------------------------------
# The dense payload
# ---------------------------------------------------------------------------

def canonical_refusals(refusals: Sequence[dict]) -> list[dict]:
    """Order diagnostic records identically for a monolith and a shard union.

    A surface refusal names a family; an incomplete sampled rung names a
    format. Both are legitimate records. The full canonical record breaks
    ties without depending on dictionary or shard insertion order.
    """
    return sorted(refusals, key=lambda entry: (
        entry["qname"], entry.get("family", ""), entry.get("format_name", ""),
        json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)))


def campaign_cost_payload(
    anchors: Mapping[str, Mapping[str, list[CampaignAnchor]]],
    menus: Mapping[str, list],
    *,
    loo: Mapping[str, Mapping[str, dict]],
    provenance: dict,
    wire_backed: "frozenset[str] | set[str]" = frozenset(),
    stack_samples: "Mapping[str, StackExpertSample] | None" = None,
) -> dict:
    """Turn measured anchors plus a legal menu into a cost payload.

    Measured rungs keep their measurement and are marked so; every other legal
    rung inside the measured envelope is interpolated and marked so; a rung
    outside the envelope is **omitted**, because ``TesseraRateSurface.predict``
    refuses to extrapolate and a menu row the surface will not price is a row
    nothing measured.

    ``wire_backed`` names the units whose priced wire IS the exported wire --
    the producer-projected packed experts, which the export lane hands to the
    exporter as cached bytes rather than re-encoding.  For those, an
    interpolated row would price a rung that has no bytes to ship, so they get
    measured rows only: the allocator can select for them exactly the rungs a
    wire exists for (priced == written; PrismaQuant #183).

    ``stack_samples`` maps a packed expert parameter's qname to the
    ``StackExpertSample`` describing which of its experts were measured.  Each
    one collapses its members' per-expert anchors into ONE stack-level row per
    (family, rung) at the packed qname -- the key the probe already has, so the
    row carries the packed topology the serving scope needs and the ``h_trace``
    the allocator will multiply.  The members are then NOT top-level cost keys
    (their measurements ride along on the stack row's ``sampled_experts``
    block), which is what stops the same experts being priced twice: once as a
    stack and once as themselves.
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
        # The content digest of the capture written for the export leg, so
        # the allocation binds to the payload and not only to the draw's
        # triple (RobTand/prismaquant#204); None on a weights-only campaign.
        "capture_sha256": _h.get("capture_sha256"),
        "kwarg": tuple(_h.get("kwargs", ())) or _h.get("kwarg"),
    }

    costs: dict[str, dict[str, dict]] = {}
    formats: set[str] = set()
    surfaces: dict[str, dict[str, TesseraRateSurface]] = {}
    refused: list[dict] = []
    # Every unit that is a MEMBER of a sampled stack, and the stack that owns
    # it. A member never becomes its own cost key: the DP would otherwise see
    # the same experts twice -- once inside the packed stack row and once as
    # standalone units -- and the union-find promotion cannot merge what it
    # cannot see as one group.
    samples: dict[str, StackExpertSample] = dict(stack_samples or {})
    member_owner: dict[str, str] = {}
    for packed_qname, sample in samples.items():
        if sample.packed_qname != packed_qname:
            raise StackSampleError(
                f"{packed_qname}: sampling record names "
                f"{sample.packed_qname!r}")
        if packed_qname in anchors:
            raise StackSampleError(
                f"{packed_qname}: the packed stack itself carries measured "
                "anchors; a stack that was measured whole is not a sample and "
                "must not also be estimated from one")
        for expert, members in sample.members.items():
            for member in members:
                owner = member_owner.setdefault(member, packed_qname)
                if owner != packed_qname:
                    raise StackSampleError(
                        f"{member}: claimed by both {owner} and "
                        f"{packed_qname}; a measured expert belongs to exactly "
                        "one stack")
    for qname, by_family in anchors.items():
        if qname in member_owner:
            continue
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
                    "encode_seconds_accounting": ("unit" if anchor.encoding_batch_size == 1
                        else "batch_wall_time_divided_by_batch_size"),
                    "encoding_batch_size": anchor.encoding_batch_size,
                    # The static A-side scale this row was scored under, when
                    # its route executes the static NVFP4 contract; the value
                    # identity the export's scale file must reproduce.
                    **({"input_global_scale": float(anchor.input_global_scale)}
                       if anchor.input_global_scale is not None else {}),
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
            if qname in wire_backed:
                # Measured rows only: see the docstring.  The surface is still
                # built so leave-one-out reporting covers these units.
                continue
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
                    # so they share a scale plane and therefore an answer --
                    # and, on a static-contract route, a unit-level A scale.
                    **({"input_global_scale": float(ordered[0].input_global_scale)}
                       if ordered[0].input_global_scale is not None else {}),
                    "hessian_identity": {
                        **hessian_identity,
                        "applied": bool(ordered[0].hessian_applied),
                    },
                }
                formats.add(rung.format_name)
        if rows:
            costs[qname] = rows
    for packed_qname, sample in sorted(samples.items()):
        stack_rows, stack_surfaces = _stack_cost_rows(
            sample, anchors, menus, hessian_identity, refused, wire_backed=wire_backed)
        if not stack_rows:
            continue
        costs[packed_qname] = stack_rows
        formats.update(stack_rows)
        if stack_surfaces:
            surfaces.setdefault(packed_qname, {}).update(stack_surfaces)
    payload = dict(provenance)
    payload.update({
        "schema": SCHEMA,
        "costs": costs,
        "formats": sorted(formats),
        "currency": CURRENCY,
        "leave_one_anchor_out": {
            q: {f: dict(v) for f, v in by_f.items()} for q, by_f in loo.items()
        },
        "non_interpolable": canonical_refusals(refused),
    })
    # The attested objective this table prices (re-vet R2; read for reuse by
    # `cost_currency.require_run_currency`, never from the environment). Every
    # row here is an `output_mse` under the route's activation contract --
    # the render-score objective -- so the stamp is unconditional: a caller
    # claim of any other mode would launder cross-currency rows into that
    # run's knapsack (RobTand/prismaquant#127).
    from .cost_currency import RENDER_SCORE_COST_MODE

    nested = payload.get("provenance")
    if not isinstance(nested, dict):
        nested = {}
        payload["provenance"] = nested
    nested["cost_mode"] = RENDER_SCORE_COST_MODE
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _wire_path(wire_dir: Path, qname: str, format_name: str) -> Path:
    return wire_dir / f"{qname.replace('.', '__')}__{format_name}.tessera"


def _checkpoint_identity_api():
    try:
        from tessera import cached_unit
    except ImportError as exc:
        raise RuntimeError(
            "Tessera campaign checkpoint identity requires the producer's "
            "cached_unit input/byte receipt API; refusing unbound resume") from exc
    return cached_unit


def _campaign_checkpoint_identity(*, weights, acts, hessians, menus, args,
                                  calibration_identity, serving_scope,
                                  static_scales, static_scale_policy,
                                  expert_projection=None, stack_sampling_identity=None):
    """Bind the priced population, including score inputs when H is off.

    The static A-side contract is a scoring input like the score rows: the
    policy is env-resolved (``PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE``) and
    the per-unit ``input_global_scale`` is a function of every calibration
    row, not of the bounded rows or the Hessian bound beside it.  Binding
    both here is what makes a checkpoint priced under another calibration or
    policy refuse at the journal (``checkpoint identity mismatch at
    units.<unit>.input_global_scale``) before any of its rows is read; the
    per-row half of the same rule lives in :func:`_checkpoint_anchor_identity`.
    """
    from .production_weight_cache import _production_cache_source_sha256

    api = _checkpoint_identity_api()
    settings = vars(args).copy()
    # Locations, batch width and a wall-clock interruption limit are not
    # encoding/scoring inputs. All other explicit campaign settings remain bound by default.
    #
    # ``units``, ``calibration_census`` and ``census_out`` are locations too,
    # and each one's load-bearing content is already bound by value somewhere
    # in this identity: the selection by the ``units`` map below (which holds
    # exactly the selected units) and ``stack_sampling_identity`` (the probe,
    # inclusion probabilities and audit draw), the census by ``calibration.fit_tokens`` /
    # ``fit_tokens_min``, and ``census_out`` by nothing, because a census run
    # writes no checkpoint. Binding the paths instead would make two shards
    # given the same selection under different filenames two identities, and
    # would make every sharded run's identity differ under two spellings of one
    # selection.  A sharded identity is still narrower than a whole-scope one --
    # the ``units`` map holds exactly the selection -- so a shard does not
    # resume a whole-scope journal; it adopts that journal's rows through
    # ``--seed-checkpoint``, one verified row at a time.
    for name in ("out", "cache_dir", "checkpoint", "deadline_seconds",
                 "units", "calibration_census", "census_out",
                 "seed_checkpoint", "seed_wire_dir", "anchor_batch_size"):
        settings.pop(name, None)
    return {
        **({"stack_sampling_identity": stack_sampling_identity}
           if stack_sampling_identity else {}),
        "campaign_schema": SCHEMA,
        "currency": CURRENCY,
        "settings": settings,
        "calibration": calibration_identity,
        "serving_scope": serving_scope,
        "encoder_recipe": th.encoder_recipe(),
        "prismaquant_source_sha256": _production_cache_source_sha256(),
        "encoder_source_sha256": api.encoder_source_sha256(),
        "input_global_scale_policy": str(static_scale_policy),
        # The producer's projection the packed units were priced under: its
        # source checkpoint identity and every sealed unit record.  A resume
        # from a campaign that projected another checkpoint, or none, refuses
        # here before a packed row is read (PrismaQuant #183).
        "expert_projection": (
            None if expert_projection is None else {
                "source": dict(expert_projection["producer"]["source"]),
                "stacks": {
                    stack: {name: dict(unit) for name, unit in sorted(units.items())}
                    for stack, units in sorted(expert_projection["stacks"].items())
                },
            }),
        "units": {
            name: {
                "weight": api.tensor_identity(weight),
                "scoring_rows": (None if acts.get(name) is None
                                 else api.tensor_identity(acts[name])),
                "hessian": (None if hessians.get(name) is None
                            else api.tensor_identity(hessians[name])),
                "input_global_scale": (
                    None if static_scales.get(name) is None
                    else float(static_scales[name])),
                "menu": sorted(rung.format_name for rung in menus[name]),
            }
            for name, weight in sorted(weights.items())
        },
    }


def _checkpoint_anchor_identity(anchor, *, weights, menus, calibration_source,
                                static_scales, projected_units=None):
    """The resumed row's inputs, as this run's producer would stamp them.

    A unit in ``projected_units`` (``{qname: producer unit record}``) is a
    packed expert the producer projects; its receipt is sealed with the
    producer's ``unit_input_identity`` -- the same encoding inputs plus the
    projection record -- because that is the identity the exporter's
    ``--cached-expert-units`` intake recomputes from the source bytes.  A
    dense unit keeps ``encoding_input_identity``.

    ONE gate per resumed row, in the order the producer resolves the inputs:
    the unit is priced, the format is a Tessera rung on the current menu, the
    Hessian applicability is what ``rung_accepts_hessian`` says for that
    rung's wire, and the A-side contract on the row is what
    :func:`_measure_anchor` stamps for that rung under this run's static
    scales (:func:`_require_resumable_anchor`).  Only then is the producer's
    ``encoding_input_identity`` asked for, so the wire receipt is verified
    against a row already known to be a price of this run.
    """
    from .tessera_formats import parse_tessera_format_name, tessera_wire_recipe
    from .tessera_render import rung_accepts_hessian

    if anchor.qname not in weights:
        raise RuntimeError(f"checkpoint anchor names an unknown unit: {anchor.qname!r}")
    parsed = parse_tessera_format_name(anchor.format_name)
    if parsed is None:
        raise RuntimeError(f"checkpoint anchor format is not Tessera: {anchor.format_name!r}")
    family, rung = parsed
    if (anchor.family, anchor.body_rate_q256) != (family.name, rung):
        raise RuntimeError("checkpoint anchor family/rung disagrees with its format")
    if anchor.format_name not in {entry.format_name for entry in menus[anchor.qname]}:
        raise RuntimeError(f"checkpoint anchor is outside the current menu: {anchor.format_name}")
    wire = tessera_wire_recipe(family, rung)
    activation = (calibration_source if calibration_source is not None
                  and rung_accepts_hessian(anchor.format_name, wire) else None)
    if bool(anchor.hessian_applied) != (activation is not None):
        raise RuntimeError("checkpoint anchor Hessian applicability disagrees with the producer")
    _require_resumable_anchor(anchor, static_scales)
    api = _checkpoint_identity_api()
    projected = (projected_units or {}).get(anchor.qname)
    if projected is not None:
        return api.unit_input_identity(
            weights[anchor.qname], dict(projected), family.payload_grid(), int(rung),
            activation=activation,
        )
    return api.encoding_input_identity(
        weights[anchor.qname], anchor.qname, family.payload_grid(), int(rung),
        activation=activation,
    )


def _checkpoint_wire_record(anchor, wire_dir, identity, *, existing=None):
    """Use the producer's one receipt grammar for fresh and resumed bytes."""
    path = _wire_path(wire_dir, anchor.qname, anchor.format_name)
    if path.is_symlink() or path.resolve().parent != wire_dir.resolve():
        raise RuntimeError(f"checkpoint cached wire escapes its directory: {path}")
    try:
        blob = path.read_bytes()
        api = _checkpoint_identity_api()
        if existing is None:
            record = api.make_unit_record(blob, identity, filename=path.name)
        else:
            record = existing
            api.verify_cached_unit(blob, record, identity)
            if record.get("file") != path.name:
                raise ValueError("cached wire filename differs from the priced unit/rung")
        # CampaignAnchor.wire_bytes means the full blob, not plane-region bytes.
        if anchor.wire_bytes != record["blob_bytes"]:
            raise ValueError("cached wire length differs from the measured anchor")
        return record
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(
            f"checkpoint cached wire identity refused for {anchor.qname} "
            f"{anchor.format_name}: {exc}") from exc


def _adopt_seed_checkpoint(manifest_path, wire_dir_arg, *, targets, wire_dir,
                           adopt, admits, identity_sha256) -> dict:
    """Offer another campaign's stored anchors to this run's row gates.

    A whole-scope campaign already priced rows this run would price again.  Its
    journal cannot be resumed as a journal -- ``prepare_journal`` binds the run
    identity, and a sharded run's identity is narrower by construction, while
    ``prismaquant_source_sha256`` moves with any change to this package -- so
    the rows are offered one at a time to the same gates a resume uses.  Those
    gates are content checks and strictly stronger than the journal's: the
    producer's ``encoding_input_identity`` is recomputed from THIS run's
    weight, menu, Hessian applicability and static scale, and Tessera's
    ``verify_cached_unit`` re-reads the blob and re-validates the wire against
    it.  A row that does not describe this run's bytes is refused by name.

    What is inherited and not re-derived is the stored ``dloss`` -- the same
    thing a resume of this run's own checkpoint inherits, for the same reason.

    Returns the record stamped into provenance: which manifest, which identity
    it was written under, and which units were adopted.
    """
    from .cost_stage_checkpoint import unit_path

    manifest = Path(manifest_path)
    parts = manifest.with_name(manifest.name + ".parts")
    if not parts.is_dir():
        raise RuntimeError(
            f"--seed-checkpoint {manifest}: no unit shards at {parts}")
    seed_wire = (Path(wire_dir_arg) if wire_dir_arg
                 else manifest.parent / "cache" / "wire")
    try:
        seed_identity = json.loads(manifest.read_text()).get("identity_sha256")
    except Exception as exc:
        raise RuntimeError(
            f"--seed-checkpoint {manifest}: unreadable manifest: {exc}") from exc
    adopted: list[str] = []
    for name in targets:
        path = unit_path(parts, name)
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                envelope = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(
                f"--seed-checkpoint {manifest}: unit shard for {name} is "
                f"unreadable: {exc}") from exc
        if not isinstance(envelope, Mapping) or envelope.get("qname") != name:
            raise RuntimeError(
                f"--seed-checkpoint {manifest}: unit shard for {name} is not "
                "an envelope for that unit")
        payload = envelope.get("payload")
        import hashlib

        if not isinstance(payload, bytes) or envelope.get("payload_sha256") != \
                hashlib.sha256(payload).hexdigest():
            raise RuntimeError(
                f"--seed-checkpoint {manifest}: unit shard for {name} fails its "
                "own payload digest")
        state = pickle.loads(payload)
        # Only the rows this run's menu admits get their bytes linked in.  An
        # unservable row's blob must not appear in this run's wire directory:
        # that directory is what the export intake reads, and a wire nothing
        # priced has no business being offered to it.
        for fmt, record in state.get("wire_records", {}).items():
            if not admits(name, fmt):
                continue
            _link_seed_wire(seed_wire, wire_dir, record.get("file"))
        adopt(name, state, where=f"seed checkpoint {manifest}")
        adopted.append(name)
    print(f"[campaign] adopted verified anchors for {len(adopted)} units from "
          f"{manifest}", flush=True)
    return {
        "manifest": str(manifest),
        "wire_dir": str(seed_wire),
        "seed_identity_sha256": seed_identity,
        "run_identity_sha256": identity_sha256,
        "units": sorted(adopted),
    }


def _link_seed_wire(seed_wire: Path, wire_dir: Path, filename) -> None:
    """Put a seed's priced wire where this run's receipt check will read it."""
    if not isinstance(filename, str) or not filename or "/" in filename:
        raise RuntimeError(f"seed wire receipt names an unusable file: {filename!r}")
    target = wire_dir / filename
    if target.exists():
        return
    source = seed_wire / filename
    if not source.is_file():
        raise RuntimeError(f"seed checkpoint has no priced wire at {source}")
    try:
        os.link(source, target)
    except OSError:
        target.write_bytes(source.read_bytes())


def _collect_activations(model, targets, tokens, max_rows: int, device,
                         *, want_hessian: bool = False, profile=None):
    """One model forward per batch, for dense and declared packed projections.

    Returns ``(rows, hessians, token_counts, max_abs)``. Counts describe every
    observed input row, including when Hessians and retained scoring rows are
    disabled; they are calibration provenance, not a Hessian-computation flag.

    Three different things come out of the same hook, and they have different
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
    * ``max_abs`` is ``max|x|`` over **every** calibration row, unconditional
      and for the same reason the Hessian is uncapped: it calibrates the
      static NVFP4 ``input_global_scale`` the W4A4 routes are priced and
      served under, and a maximum over a prefix of the rows is a different
      calibration than the one an exporter would take.

    Accumulated in fp32 on the model's device and moved to CPU once at the
    end, matching how the kept rows are handled. Packed units use the existing
    module-input collector and routing/SwiGLU derivation, and land in the SAME
    three accumulators as a dense Linear -- rows, Hessian and ``max_abs`` --
    so there is one notion of "what the campaign captured" for either
    population, and ``write_export_inputs`` has one input to write.  The
    score-row cap never caps routed rows before the Hessian or the maximum.
    The main-entry gate requires profile-declared projections and the
    producer's planning tool before calibration. The campaign later binds
    that producer projection to these live views before pricing the units
    and carrying their wire receipts into export (PrismaQuant #183).
    """
    import torch

    store: dict[str, list] = {name: [] for name in targets}
    kept: dict[str, int] = {name: 0 for name in targets}
    hess: dict[str, object] = {name: None for name in targets}
    seen: dict[str, int] = {name: 0 for name in targets}
    amax: dict[str, float] = {name: 0.0 for name in targets}
    routed_seen: dict[str, int] = {}
    handles = []
    packed_collector = None

    def accumulate(name, x):
        # ONE accumulator for both populations: a dense Linear's pre-hook and
        # a packed projection's routed rows land here, so the three outputs
        # (score rows, Hessian, max|x|) are the same three for either, and a
        # packed unit can feed ``_static_input_scales``/``write_export_inputs``
        # by the same path a dense one does once the main-entry packed gate
        # opens.  A zero-row call (an expert no token was routed to on this
        # batch) contributes nothing to any of them.
        flat = x.detach().reshape(-1, x.shape[-1])
        if not flat.shape[0]:
            return
        if name in routed_seen:
            routed_seen[name] += int(flat.shape[0])
        amax[name] = max(
            amax[name], float(flat.abs().amax().float().item()))
        seen[name] += int(flat.shape[0])
        if want_hessian:
            # Every row, before any cap: see the docstring.
            f32 = flat.to(dtype=torch.float32)
            gram = f32.t() @ f32
            if hess[name] is None:
                hess[name] = gram
            else:
                hess[name] += gram
        room = max_rows - kept[name]
        if room <= 0:
            return
        take = flat[:room].to(dtype=torch.float32, device="cpu")
        store[name].append(take)
        kept[name] += int(take.shape[0])

    def make_hook(name):
        def hook(_module, args):
            if not args:
                return
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            accumulate(name, x)
        return hook

    modules = dict(model.named_modules())
    missing_modules = set(targets) - modules.keys()
    if missing_modules:
        from .routed_experts import (
            profile_declared_packed_expert_projections,
            resolve_routed_expert_profile,
        )
        from .measure_quant_cost import (
            _packed_experts_parent_module, derive_per_expert_activations,
        )
        from .production_weight_cache import _PackedExpertActivationCollector

        profile = resolve_routed_expert_profile(model, profile)
        inventory = profile_declared_packed_expert_projections(model, profile)
        by_name = {member.qname: member for member in inventory}
        unknown = missing_modules - by_name.keys()
        if unknown:
            raise RuntimeError(f"campaign activation targets are not declared units: {sorted(unknown)}")
        selected_by_module = {}
        # These are the input kinds published by the existing derivation,
        # not a mapping to the producer's served role/group vocabulary.
        input_kind = {"gate_up_proj": "gate_up", "down_proj": "down"}
        for name in sorted(missing_modules):
            member = by_name[name]
            if member.param_name not in input_kind:
                raise RuntimeError(
                    f"packed activation derivation does not support {member.packed_qname!r}")
            selected_by_module.setdefault(member.module_qname, []).append(member)
            routed_seen[name] = 0
        parents = {name: _packed_experts_parent_module(model, name)
                   for name in selected_by_module}

        def consume_packed(module_qname, x):
            selected = selected_by_module.get(module_qname)
            if not selected:
                return
            derived = derive_per_expert_activations(
                selected[0].module, x, parents[module_qname],
                capture_down=any(input_kind[member.param_name] == "down"
                                 for member in selected),
                max_rows_per_expert=None,
            )
            for member in selected:
                accumulate(member.qname,
                           derived[input_kind[member.param_name]][member.expert_id])

        packed_collector = _PackedExpertActivationCollector(
            model, {member.module_qname for member in inventory},
            module_token_budget=0, store_device=device, store_qnames=set(),
            profile=profile, row_consumer=consume_packed,
        )
    try:
        for name in targets:
            if name not in missing_modules:
                handles.append(modules[name].register_forward_pre_hook(make_hook(name)))
        if packed_collector is not None:
            packed_collector.install()
        with torch.no_grad():
            for batch in tokens:
                model(batch.to(device))
    finally:
        for handle in handles:
            handle.remove()
        if packed_collector is not None:
            packed_collector.remove()
    unobserved = [name for name, count in routed_seen.items() if not count]
    if unobserved:
        raise RuntimeError(
            "packed campaign units have no routed calibration rows: "
            f"{sorted(unobserved)}; refusing a shared-Hessian or weight-only fallback")
    # Do not retain both the original chunks and their concatenated outputs
    # for the whole scope. Full-model captures can hold GiB of scoring rows.
    rows = {}
    for name in list(store):
        chunks = store.pop(name)
        rows[name] = torch.cat(chunks, dim=0) if chunks else None
        del chunks
    hessians = {}
    if want_hessian:
        # GB10 shares physical DRAM with CPU tensors. Releasing Python tensor
        # references alone leaves the CUDA allocator's cached blocks resident,
        # so periodically return those blocks while transferring the full H.
        released_cuda_bytes = 0
        for name in list(hess):
            h = hess.pop(name)
            hessians[name] = None if h is None else h.to(device="cpu")
            if h is not None and h.is_cuda:
                released_cuda_bytes += h.numel() * h.element_size()
            del h
            if released_cuda_bytes >= 256 * 1024**2:
                torch.cuda.empty_cache()
                released_cuda_bytes = 0
        if released_cuda_bytes:
            torch.cuda.empty_cache()
    return rows, hessians, dict(seen), dict(amax)


def _static_input_scales(max_abs: "Mapping[str, float]", *, profile=None):
    """Per-unit static NVFP4 ``input_global_scale``, fused-sibling unified.

    ``(scales, policy)``.  Everything here is the owned NVFP4 activation
    contract, reused rather than restated: the resolved default policy names
    the formula (``resolve_input_global_scale_policy``), the scalar is the
    F32-rounded value an exported tensor would carry
    (``input_global_scale_from_max_abs``), and fused siblings share one
    conservative calibration maximum (``unify_fused_sibling_max_abs``) --
    vLLM concatenates q/k/v and gate/up and applies ONE activation scale, and
    Tessera's exporter joins its members' scales for the same module, so a
    per-member scale would price an A side no fused module executes.

    A unit whose calibration never saw a row keeps no scale; a W4A4 anchor on
    it then refuses in :func:`_measure_anchor` rather than pricing dynamically.
    """
    from .nvfp4_activation_contract import (
        input_global_scale_from_max_abs, resolve_input_global_scale_policy,
        unify_fused_sibling_max_abs,
    )

    policy = resolve_input_global_scale_policy()
    positive = {name: float(value) for name, value in max_abs.items()
                if float(value) > 0.0}
    unified = unify_fused_sibling_max_abs(
        positive, profile=profile, tolerate_profile_errors=True)
    return {
        name: input_global_scale_from_max_abs(value, policy=policy)
        for name, value in unified.items()
    }, policy


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


#: The exit status a run uses when the resolved menu admits nothing.  It is
#: not 1: a menu that admits nothing is a configuration answer, not a crash,
#: and a caller that fans out over hundreds of rows needs to tell the two
#: apart without parsing a traceback.
EXIT_EMPTY_MENU = 2


def contract_source_label() -> str:
    """Which Tessera contract the menu was resolved against, in one phrase.

    A refusal has to say this or it is unactionable: ``attested`` resolves to
    nothing at all unless the dev pin names a commit, and the difference
    between "the pinned reader admits no rung of this shape" and "no contract
    was consulted" is the whole diagnosis.
    """
    import os

    from .tessera_runtime_contract import TESSERA_DEV_PIN_ENV

    pin = str(os.environ.get(TESSERA_DEV_PIN_ENV, "")).strip()
    if pin:
        return f"{TESSERA_DEV_PIN_ENV}={pin}"
    return "the packaged Tessera runtime_contract.json (no dev pin set)"


def report_empty_menus(menus: "Mapping[str, Sequence]", *, mode: str) -> list[str]:
    """The units with no admitted rung, printed with the mode and the contract.

    Printed rather than raised, because one unit whose shape admits nothing is
    a fact about that unit and the rest of the run is still a real answer --
    it is stamped ``no_admitted_rung`` so a merge cannot mistake a refused
    unit for one nobody selected.  Only the *total* emptiness is fatal, and
    that decision is the caller's (:data:`EXIT_EMPTY_MENU`).
    """
    empty = sorted(name for name, menu in menus.items() if not menu)
    if not empty:
        return []
    print(f"[campaign] {len(empty)} of {len(menus)} units admit no rung under "
          f"mode={mode} against {contract_source_label()}:", flush=True)
    for name in empty:
        print(f"[campaign]   menu_sizes[{name}] = 0", flush=True)
    return empty


def expand_menus_for_targets(weights, targets, *, mode, tp_degree,
                             parallel_kind,
                             context_by_unit: "Mapping[str, ServingContext] | None" = None,
                             ) -> dict[str, list]:
    """One Tessera menu per distinct shape and explicit serving context.

    ``expand_tessera_menu`` takes nothing but the shape and the run
    configuration and serving context, so units with equal values of those
    inputs get identical lists. Dense and routed expert units may share one
    shape; their structural class comes from context_by_unit, never shape or
    name. A missing map entry remains unbound. Units
    repeat shapes ~1500:1 on a production MoE, so expanding per Linear repeats
    the same answer thousands of times; keying by shape and context expands once per
    distinct answer instead.  Exact rather than approximate: same arguments,
    same list.  The lists are shared, not copied -- downstream only iterates
    them -- which is also what makes ``menu_cache_shapes``' retention the
    thing that bounds the work.
    """
    from .tessera_menu import expand_tessera_menu

    by_shape_and_context: dict[tuple, list] = {}
    menus: dict[str, list] = {}
    for name in targets:
        shape = tuple(weights[name].shape)
        context = None if context_by_unit is None else context_by_unit.get(name)
        key = (shape, None if context is None else context.key())
        if key not in by_shape_and_context:
            by_shape_and_context[key] = expand_tessera_menu(
                shape, mode=mode, tp_degree=tp_degree,
                parallel_kind=parallel_kind,
                **({"serving_context": context} if context is not None else {}),
            )
        menus[name] = by_shape_and_context[key]
    return menus


#: The selection file ``--units`` reads and ``tools/dispatch_tessera_campaign.py``
#: writes.  A selection names **fused anchor groups**, never bare units: anchor
#: placement is a group property (one shared rung grid, the group's worst
#: member drives the split), so a group is the smallest scope whose measured
#: values do not depend on what else the run priced.
UNITS_SCHEMA = "prismaquant.tessera_campaign_units.v1"

#: The same selection, plus the planner's expert sample.  A ``v1`` file prices
#: every member of every group it names; a ``v2`` file may additionally give a
#: group a ``sampled`` subset (with the ``inclusion_probability`` it was drawn
#: under and the ``audit`` units inside it).  The group's ``members`` list
#: still names the WHOLE group in both, because that list is what
#: :func:`select_anchor_groups` checks against this run's own grouping: the
#: sample says which members are priced, never which members exist.  A run
#: given a v1 file behaves exactly as it did before v2.
UNITS_SCHEMA_V2 = "prismaquant.tessera_campaign_units.v2"

#: The calibration census ``--calibration-census`` reads and ``--census-out``
#: writes: the per-unit calibration row counts of the whole priced scope.
CENSUS_SCHEMA = "prismaquant.tessera_campaign_census.v1"


def _keystream(seed: int, label: str):
    """A stable, dependency-free stream of uniforms in [0, 1).

    SHA-256 in counter mode rather than :mod:`random`, because the draw is
    stamped into a checkpoint identity: a sample that changed when CPython
    changed its sampler would silently re-key every journal it appears in.
    """
    import hashlib

    counter = 0
    while True:
        digest = hashlib.sha256(
            f"{int(seed)}:{label}:{counter}".encode()).digest()
        for offset in range(0, 32, 8):
            yield int.from_bytes(digest[offset:offset + 8], "big") / float(1 << 64)
        counter += 1


def _permute(names: "Sequence[str]", stream) -> list[str]:
    """Fisher-Yates over ``names`` from ``stream``."""
    order = list(names)
    for index in range(len(order) - 1, 0, -1):
        pick = int(next(stream) * (index + 1))
        pick = min(pick, index)
        order[index], order[pick] = order[pick], order[index]
    return order


def draw_stack_sample(weights: "Mapping[str, float]", n: int, *,
                      seed: int, stack: str) -> dict:
    """A fixed-size PPS draw over one stack's experts, with exact ``pi``.

    Randomized systematic (Madow) probability-proportional-to-size sampling
    without replacement, with a take-all stratum.  Three parts, each of which
    is load-bearing:

    **Proportional to what.** The stack total the allocator needs is
    ``sum_e h_e * mse_e``.  ``h_e`` is free (the probe already has it) and
    ``mse_e`` costs an encode, so drawing proportional to ``h_e`` leaves only
    the spread of ``mse_e`` in the estimator's variance.  On LFM2.5 layer 18
    the per-expert ``h_trace`` spans 9.6e4 to 7.7e6 (CV ~1.0) while the
    cross-expert spread of ``mse`` at a fixed rung is CV 0.33-0.55, so this is
    where the variance is. A uniform draw with the correct Horvitz-Thompson
    weights remains unbiased, but carries the product's spread. The planner
    requires Fisher weights and provides no uniform fallback.

    **The take-all stratum.** With ``h`` this dispersed, ``n * h_i / sum(h)``
    exceeds one for the largest experts.  Clipping that to one would leave
    ``sum(pi) < n`` and bias the estimate low.  Instead those experts are
    taken with certainty, removed from the frame, and the rest re-solved --
    iterated to a fixed point, which is ``pi_i = min(1, c*h_i)`` for the
    unique ``c`` with ``sum_i min(1, c*h_i) = n``.  The post-conditions
    asserted below (``sum(pi) == n``, every ``pi`` in ``[0, 1]``) are what
    make the Horvitz-Thompson estimator downstream exactly unbiased rather
    than approximately so.

    **The randomized order.** The frame is permuted from the seeded stream
    before the systematic pass.  This is not cosmetic: the practical variance
    estimator for this design (Hartley-Rao, which #290 applies to the prices)
    is derived for randomized-order systematic sampling and is not justified
    under expert-index or ``h``-sorted order.

    Determinism, and what the seed is deliberately NOT: the stream is keyed on
    ``(seed, stack)`` only, never on ``h``.  The probe is not bit-reproducible
    across runs, and systematic sampling is continuous in ``pi``, so a
    re-probed ``h`` perturbs this draw slightly; hashing ``h`` into the seed
    would instead redraw the whole stack on every re-probe.

    An expert with ``h_e == 0`` gets ``pi_e = 0`` and is never encoded, which
    is exact rather than a rounding: under the campaign's Fisher convention a
    token never routed to an expert contributes zero gradient, so its term in
    the total is zero, not merely small.

    Returns the whole draw, including everything a variance estimate needs:
    ``units``, ``inclusion_probability`` (over the full frame), ``certainty``,
    ``permutation``, ``start``, ``size``, ``size_sha256``, ``frame_size``,
    ``random_draws`` and ``method``.
    """
    import hashlib

    names = sorted(weights)
    if not names:
        raise RuntimeError(f"stack {stack}: no unit to sample")
    sizes = {name: float(weights[name]) for name in names}
    if any(not math.isfinite(value) for value in sizes.values()):
        raise RuntimeError(f"stack {stack}: size weights must be finite")
    if any(value < 0.0 for value in sizes.values()):
        raise RuntimeError(f"stack {stack}: a size weight is negative")
    digest = hashlib.sha256(
        "|".join(f"{name}={sizes[name]!r}" for name in names).encode()
    ).hexdigest()
    want = int(n)
    if want < 1:
        raise RuntimeError(f"stack {stack}: --stack-sample must be at least 1")

    frame = [name for name in names if sizes[name] > 0.0]
    zero = [name for name in names if sizes[name] <= 0.0]
    common = {"size": sizes, "size_sha256": digest, "seed": int(seed),
              "stack": str(stack), "frame_size": len(frame),
              "zero_size": zero}
    if want >= len(frame):
        return {**common, "units": list(frame), "method": "census",
                "inclusion_probability": {name: (1.0 if sizes[name] > 0.0 else 0.0)
                                          for name in names},
                "certainty": list(frame), "permutation": list(frame),
                "start": None, "random_draws": 0}

    certainty: list[str] = []
    rest = list(frame)
    remaining = want
    while rest and remaining > 0:
        total = sum(sizes[name] for name in rest)
        over = [name for name in rest if remaining * sizes[name] / total >= 1.0]
        if not over:
            break
        certainty.extend(over)
        rest = [name for name in rest if name not in set(over)]
        remaining -= len(over)
    certainty.sort()

    # A single random draw has no variance estimate. Stamping 0.0 for it would
    # read downstream as "measured exactly", so it refuses here instead --
    # before any GPU second is spent on a sample nothing can put an error bar
    # on. Either take one more expert, or take the stack whole.
    if remaining == 1:
        raise RuntimeError(
            f"stack {stack}: --stack-sample {want} leaves exactly one "
            f"randomly drawn expert after {len(certainty)} certainty unit(s); "
            "a one-draw sample admits no variance estimate. Use "
            f"--stack-sample {want + 1} or price the stack whole.")

    pi = {name: 0.0 for name in names}
    pi.update({name: 1.0 for name in certainty})
    drawn = list(certainty)
    stream = _keystream(seed, f"pps:{stack}")
    permutation = _permute(rest, stream) if rest else []
    start = None
    if remaining > 0 and rest:
        total = sum(sizes[name] for name in rest)
        pi.update({name: remaining * sizes[name] / total for name in rest})
        start = next(stream)
        cumulative = 0.0
        ladder = []
        for name in permutation:
            cumulative += pi[name]
            ladder.append((cumulative, name))
        # Renormalise the last edge so float drift cannot drop the final draw.
        ladder[-1] = (float(remaining), ladder[-1][1])
        step = 0
        for edge, name in ladder:
            while step < remaining and start + step < edge:
                drawn.append(name)
                step += 1
    total_pi = sum(pi.values())
    if abs(total_pi - want) > 1e-9:
        raise RuntimeError(
            f"stack {stack}: inclusion probabilities sum to {total_pi!r}, not "
            f"{want}; the estimator built on them would be biased")
    if len(set(drawn)) != want:
        raise RuntimeError(
            f"stack {stack}: the systematic pass drew {len(set(drawn))} "
            f"distinct units, not {want}")
    return {**common, "units": sorted(drawn),
            "method": "randomized_systematic_pps_with_take_all_v1",
            "inclusion_probability": {name: float(pi[name]) for name in names},
            "certainty": certainty, "permutation": permutation,
            "start": start, "random_draws": int(remaining)}


def audit_subsample(drawn: "Sequence[str]", *, rate: int = 10, seed: int = 0,
                    stack: str = "") -> list[str]:
    """A simple random subsample of the draw: one unit in ``rate``, at least 1.

    Drawn from a stream of its own, so changing the audit fraction cannot
    disturb the sample the prices are built from -- the audit buys evidence
    about interpolation, and must not be able to move the estimate.
    """
    units = sorted(drawn)
    if not units:
        return []
    count = max(1, len(units) // int(rate))
    stream = _keystream(seed, f"audit:{stack}")
    return sorted(_permute(units, stream)[:count])


def anchor_group_key(name: str, *, profile, expert_members: Mapping) -> str:
    """The fused anchor group ``name`` belongs to.

    A projected expert unit anchors with its whole stack: the producer plans
    ONE rung per stack, so every member must measure the same rungs for the
    allocator's stack-uniform choice to have a priced (and wire-backed) row on
    every member.  A dense unit anchors with its fused siblings, for the reason
    the round loop states: the grid is shared, so a rung added for one sibling
    is measured for all of them anyway.
    """
    member = expert_members.get(name)
    if member is not None:
        return f"s:{member.module_qname}"
    try:
        key = profile.fused_sibling_group(name)
    except Exception:
        key = None
    return f"g:{key}" if key else f"u:{name}"


def resolve_anchor_groups(targets: Sequence[str], *, profile,
                          expert_members: Mapping) -> dict[str, list[str]]:
    """``{group key: sorted members}`` over ``targets``."""
    groups: dict[str, list[str]] = {}
    for name in targets:
        groups.setdefault(
            anchor_group_key(name, profile=profile, expert_members=expert_members),
            []).append(name)
    for key in groups:
        groups[key].sort()
    return groups


def load_unit_selection(path) -> dict:
    """Read a ``--units`` selection file, refusing anything but this schema."""
    selection = json.loads(Path(path).read_text())
    schema = None if not isinstance(selection, dict) else selection.get("schema")
    if schema not in (UNITS_SCHEMA, UNITS_SCHEMA_V2):
        raise RuntimeError(
            f"--units {path}: not a {UNITS_SCHEMA} or {UNITS_SCHEMA_V2} "
            "selection")
    groups = selection.get("groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError(f"--units {path}: names no anchor group")
    for entry in groups:
        if (not isinstance(entry, dict) or not isinstance(entry.get("key"), str)
                or not isinstance(entry.get("members"), list)
                or not entry["members"]
                or not all(isinstance(m, str) for m in entry["members"])):
            raise RuntimeError(
                f"--units {path}: a group entry is not {{key, members[]}}")
        if schema == UNITS_SCHEMA and any(
                field in entry for field in ("sampled", "audit",
                                             "inclusion_probability", "stack_samples")):
            raise RuntimeError(
                f"--units {path}: group {entry['key']!r} carries a sample, "
                f"which is a {UNITS_SCHEMA_V2} field; a file that samples "
                "must say so in its schema")
        members = set(entry["members"])
        sampled = entry.get("sampled")
        if sampled is not None:
            if (not isinstance(sampled, list) or not sampled
                    or not all(isinstance(m, str) for m in sampled)
                    or not set(sampled) <= members):
                raise RuntimeError(
                    f"--units {path}: group {entry['key']!r} samples units "
                    "that are not its members")
            audit = entry.get("audit") or []
            if not isinstance(audit, list) or not set(audit) <= set(sampled):
                raise RuntimeError(
                    f"--units {path}: group {entry['key']!r} audits units it "
                    "did not sample")
            pi = entry.get("inclusion_probability")
            if not isinstance(pi, dict) or not set(sampled) <= set(pi):
                raise RuntimeError(
                    f"--units {path}: group {entry['key']!r} samples without "
                    "an inclusion probability for every sampled unit; an "
                    "unbiased estimate downstream is impossible without it")
        elif entry.get("audit"):
            raise RuntimeError(
                f"--units {path}: group {entry['key']!r} audits without "
                "sampling")
    return selection


def selection_stack_samples(selection: Mapping, profile) -> dict[str, StackExpertSample]:
    """Rehydrate packed probe/draw records and bind them to the whole group.

    A selection may narrow measured experts; it may not invent projections,
    lose an expert from the frame, or give two roles different expert draws.
    Legacy unsampled selections retain their existing per-unit behavior.
    """
    result = {}
    for entry in selection["groups"]:
        records = entry.get("stack_samples")
        if records is None:
            if entry.get("sampled") and str(entry["key"]).startswith("s:"):
                raise StackSampleError(
                    f"{entry['key']}: sampled stack requires original packed probe/draw records")
            continue
        if not isinstance(records, Mapping) or not records:
            raise StackSampleError(f"{entry['key']}: stack_samples must name packed parameters")
        frame_members, sampled_members, frame_pi, audit_members = set(), set(), {}, set()
        for name, record in sorted(records.items()):
            if name in result:
                raise StackSampleError(f"{name}: duplicate packed sampling record")
            sample = stack_sample_from_probe(
                name, record["probe_row"], profile,
                sampled_experts=record["sampled_experts"],
                inclusion_prob=record["inclusion_prob"], seed=record["seed"],
                design=record["design"])
            _validate_stack_sample(sample)
            replay = draw_stack_sample(
                {str(e): h for e, h in enumerate(sample.h_trace_per_expert)},
                len(sample.sampled_experts), seed=sample.seed, stack=name)
            if (record.get("draw") != replay or sample.design != replay["method"]
                    or list(sample.sampled_experts) != sorted(int(e) for e in replay["units"])
                    or dict(sample.inclusion_prob) != {
                        int(e): p for e, p in replay["inclusion_probability"].items()}):
                raise StackSampleError(f"{name}: packed draw receipt does not replay from probe and seed")
            if set(sample.inclusion_prob) != set(range(sample.num_experts)):
                raise StackSampleError(f"{name}: selection needs full-frame inclusion probabilities")
            if entry["key"] != "s:" + sample.packed_experts_module:
                raise StackSampleError(f"{name}: packed module disagrees with anchor group")
            frame = stack_sample_from_probe(
                name, record["probe_row"], profile,
                sampled_experts=range(sample.num_experts),
                inclusion_prob={e: 1.0 for e in range(sample.num_experts)},
                seed=sample.seed, design="census")
            for e, members in frame.members.items():
                for member in members:
                    if member in frame_members:
                        raise StackSampleError(f"{member}: two packed parameters claim one projection")
                    frame_members.add(member)
                    frame_pi[member] = sample.inclusion_prob[e]
                    if e in record.get("audit_experts", []):
                        audit_members.add(member)
            sampled_members.update(m for members in sample.members.values() for m in members)
            result[name] = sample
        if frame_members != set(entry["members"]):
            raise StackSampleError(f"{entry['key']}: packed probe projections disagree with census members")
        if sampled_members != set(entry.get("sampled", entry["members"])):
            raise StackSampleError(f"{entry['key']}: packed draw disagrees with sampled members")
        if audit_members != set(entry.get("audit", [])):
            raise StackSampleError(f"{entry['key']}: packed audit disagrees with audit members")
        if frame_pi != entry.get("inclusion_probability"):
            raise StackSampleError(f"{entry['key']}: packed draw disagrees with full-frame probabilities")
    return result


def selection_priced_units(selection: Mapping) -> tuple[set, set, dict]:
    """``(priced, audit, inclusion_probability)`` a selection asks for.

    ``priced`` is the sample where a group has one and the whole group where
    it does not, so a v1 selection and an unsampled v2 selection are the same
    run.  Nothing here decides what a group *is*: that check has already been
    made against this run's own grouping.
    """
    priced: set = set()
    audit: set = set()
    pi: dict = {}
    for entry in selection["groups"]:
        sampled = entry.get("sampled")
        priced.update(sampled if sampled else entry["members"])
        audit.update(entry.get("audit") or ())
        for name, value in (entry.get("inclusion_probability") or {}).items():
            pi[str(name)] = float(value)
    return priced, audit, pi


def select_anchor_groups(selection: Mapping, resolved: Mapping[str, list[str]],
                         *, where: str) -> list[str]:
    """The selected group keys, refusing any disagreement with this run's scope.

    The selection is checked against the grouping **this run** resolved, member
    for member.  A plan written against another model, stride or profile names
    a group whose membership differs here, and a shard that silently measured a
    different set would leave the merged table short of rows nothing reported.
    """
    keys: list[str] = []
    for entry in selection["groups"]:
        key = entry["key"]
        members = sorted(entry["members"])
        if key not in resolved:
            raise RuntimeError(
                f"{where}: selection names anchor group {key!r}, which this "
                f"run's scope does not contain")
        if sorted(resolved[key]) != members:
            raise RuntimeError(
                f"{where}: anchor group {key!r} has members "
                f"{sorted(resolved[key])} here and {members} in the selection")
        if key in keys:
            raise RuntimeError(f"{where}: anchor group {key!r} selected twice")
        keys.append(key)
    return keys


def calibration_census(counts: Mapping[str, int], max_abs: Mapping[str, float], *,
                       args, groups: Mapping, dense_targets: Sequence[str],
                       expert_targets: Sequence[str], shapes: Mapping,
                       identity: Mapping, expert_projection=None) -> dict:
    """The whole priced scope, as one run that loaded the model saw it.

    Everything here is scope-wide and selection-independent, and it is here
    precisely so that no shard has to re-derive it and no two shards can
    disagree about it: the per-unit calibration row counts and activation
    maxima, the anchor grouping, the unit shapes a planner sizes rows with, the
    draw's own identity, and the producer's expert projection of every declared
    stack.
    """
    return {
        "schema": CENSUS_SCHEMA,
        "model": str(args.model),
        "nsamples": int(args.nsamples),
        "seqlen": int(args.seqlen),
        "seed": int(args.seed),
        "layer_stride": int(args.layer_stride),
        # The draw itself, in Tessera's own vocabulary: a census taken on a
        # different corpus or tokenizer is refused before any GPU is spent.
        "text_sha256": str(identity["text_sha256"]),
        "fit_ids_sha256": str(identity["fit_ids_sha256"]),
        "counts": {str(name): int(value) for name, value in sorted(counts.items())},
        "max_abs": {str(name): float(value) for name, value in sorted(max_abs.items())},
        "unit_shapes": {str(name): [int(dim) for dim in shape]
                        for name, shape in sorted(shapes.items())},
        "anchor_groups": {str(key): sorted(members)
                          for key, members in sorted(groups.items())},
        "dense_targets": sorted(dense_targets),
        "expert_targets": sorted(expert_targets),
        "expert_projection": expert_projection,
    }


def load_calibration_census(path, *, args) -> dict:
    """Read a census and refuse one taken on a different draw or scope."""
    census = json.loads(Path(path).read_text())
    if not isinstance(census, dict) or census.get("schema") != CENSUS_SCHEMA:
        raise RuntimeError(
            f"--calibration-census {path}: not a {CENSUS_SCHEMA} census")
    for field, value in (("model", str(args.model)),
                         ("nsamples", int(args.nsamples)),
                         ("seqlen", int(args.seqlen)),
                         ("seed", int(args.seed)),
                         ("layer_stride", int(args.layer_stride))):
        if census.get(field) != value:
            raise RuntimeError(
                f"--calibration-census {path}: {field} is {census.get(field)!r} "
                f"in the census and {value!r} in this run; the census must be "
                "the same draw over the same scope")
    counts = census.get("counts")
    maxima = census.get("max_abs")
    if not isinstance(counts, dict) or not counts:
        raise RuntimeError(f"--calibration-census {path}: names no unit count")
    if not isinstance(maxima, dict) or set(maxima) != set(counts):
        raise RuntimeError(
            f"--calibration-census {path}: its activation maxima do not cover "
            "exactly the units it counted")
    census["counts"] = {str(name): int(value) for name, value in counts.items()}
    census["max_abs"] = {str(name): float(value) for name, value in maxima.items()}
    return census


def require_census_draw(census: Mapping, identity: Mapping, *, where: str) -> None:
    """Refuse a census taken on another draw than this run's.

    ``load_calibration_census`` compares the arguments that *name* a draw;
    this compares the draw itself, which is the thing the identity is of. The
    two are different checks: the same ``--seed 0 --nsamples 32`` over a
    different corpus revision is a different calibration with identical flags.
    """
    for field in ("text_sha256", "fit_ids_sha256"):
        if str(census.get(field)) != str(identity[field]):
            raise RuntimeError(
                f"{where}: the census's {field} is {census.get(field)!r} and "
                f"this run's is {identity[field]!r}; they are different draws")


def census_token_counts(census: "Mapping | None", observed: Mapping[str, int]):
    """``(max, min)`` calibration rows over the priced scope, verified.

    Without a census this is the run's own observation, which is what a
    whole-scope campaign measures.  With one it is the **scope's** maximum and
    minimum, and every unit this run actually hooked is checked against the
    census row for it first: the census is a measurement another invocation of
    this same stage published, and a shard that disagrees with it about its own
    units is not measuring the draw the census describes.  That check is what
    makes a sharded campaign's ``fit_tokens`` the whole scope's -- and therefore
    equal to the monolith's -- without any shard asserting a count it did not
    see (principle 14).
    """
    if census is None:
        if not observed:
            return 0, 0
        return max(observed.values()), min(observed.values())
    counts = census["counts"]
    missing = sorted(set(observed) - set(counts))
    if missing:
        raise RuntimeError(
            "calibration census does not cover units this run priced: "
            + ", ".join(missing))
    disagree = sorted(name for name, value in observed.items()
                      if int(counts[name]) != int(value))
    if disagree:
        raise RuntimeError(
            "calibration census disagrees with this run's observed rows for "
            + ", ".join(f"{name} (census {counts[name]}, observed {observed[name]})"
                        for name in disagree))
    return max(counts.values()), min(counts.values())


def census_max_abs(census: Mapping, observed: Mapping[str, float]) -> dict[str, float]:
    """The scope's calibration maxima, with this run's own units verified.

    Same rule as :func:`census_token_counts`, for the other unconditional
    output of the same hook: the value used is the scope's, and a run whose own
    observation disagrees with the census about a unit it hooked is refusing,
    not adopting.
    """
    maxima = census["max_abs"]
    missing = sorted(set(observed) - set(maxima))
    if missing:
        raise RuntimeError(
            "calibration census has no activation maximum for units this run "
            "priced: " + ", ".join(missing))
    disagree = sorted(name for name, value in observed.items()
                      if float(maxima[name]) != float(value))
    if disagree:
        raise RuntimeError(
            "calibration census disagrees with this run's observed activation "
            "maximum for " + ", ".join(
                f"{name} (census {maxima[name]!r}, observed {observed[name]!r})"
                for name in disagree))
    return {str(name): float(value) for name, value in maxima.items()}


def _campaign_layer_scope(names, layer_stride: int) -> list[str]:
    """The same explicit layer scope for supported and unsupported units."""
    if layer_stride <= 1:
        return list(names)
    import re

    selected = []
    for name in names:
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match is None or int(match.group(1)) % layer_stride == 0:
            selected.append(name)
    return selected


@dataclass(frozen=True)
class ExpertPopulation:
    """The packed expert population the bridge covers, and what it omits.

    ``members`` are the profile-declared per-expert projections inside the
    layer scope (``PackedExpertProjection``: the live 2-D view, its packed
    parent and its stack); ``declared`` is ``{stack: {qname: (rows, cols)}}``
    for the producer request; ``omitted_outside_layer_stride`` names the
    packed parameters the stride left out, so the payload can say which
    population was priced and which was not.
    """

    members: tuple
    declared: dict
    packed_in_scope: dict
    omitted_outside_layer_stride: dict

    @property
    def qnames(self) -> list[str]:
        return [member.qname for member in self.members]


def _require_campaign_population(model, profile, layer_stride: int) -> ExpertPopulation:
    """Admit only the packed units the producer bridge can carry; refuse the rest.

    The profile's routed-target discovery owns membership. An arbitrary 3-D
    parameter (for example a convolution) is not an expert by shape alone.
    The bridge covers a packed parameter when the profile declares its split
    into per-expert 2-D projections whose input kind the existing routed
    derivation captures (``gate_up`` / ``down``) and when the producer's
    projection tool is declared and present -- checked here, before an hour
    of calibration, so a campaign that cannot ask the producer refuses by
    name instead of pricing a dense-only table (PrismaQuant #183).
    """
    from .routed_experts import (
        profile_declared_packed_expert_projections,
        profile_declared_routed_expert_targets,
    )
    from .tessera_expert_projection import (
        ExpertProjectionError, declared_stacks_from_members, producer_plan_tool,
    )

    parameters = dict(model.named_parameters())
    packed = {
        name: tuple(parameters[name].shape)
        for name in profile_declared_routed_expert_targets(model, profile)
        if name in parameters and parameters[name].ndim == 3
    }
    in_scope = _campaign_layer_scope(packed, layer_stride)
    omitted = {name: packed[name] for name in packed if name not in set(in_scope)}
    if not in_scope:
        return ExpertPopulation(members=(), declared={}, packed_in_scope={},
                                omitted_outside_layer_stride=omitted)
    members = [
        member for member in profile_declared_packed_expert_projections(model, profile)
        if member.packed_qname in set(in_scope)
    ]
    covered = {member.packed_qname for member in members}
    # The same input-kind roster ``_collect_activations`` derives from; a
    # packed parameter the derivation cannot feed is one the bridge does not
    # cover, and it is named here rather than discovered mid-capture.
    supported_kinds = {"gate_up_proj", "down_proj"}
    uncovered = sorted(
        name for name in in_scope
        if name not in covered or name.rsplit(".", 1)[-1] not in supported_kinds)
    if uncovered:
        raise RuntimeError(
            "Tessera campaign cannot price the packed expert population: "
            + ", ".join(f"{name} {packed[name]}" for name in uncovered)
            + ". The producer bridge covers only profile-declared gate_up/down "
            "splits into per-expert 2-D projections; refusing an incomplete "
            "dense-only cost payload (PrismaQuant #183).")
    try:
        producer_plan_tool()
    except ExpertProjectionError as exc:
        raise RuntimeError(
            "Tessera campaign cannot ask the producer for its expert projection, "
            f"so the packed population {sorted(in_scope)} cannot be priced: {exc}. "
            "Refusing an incomplete dense-only cost payload (PrismaQuant #183).") from exc
    return ExpertPopulation(
        members=tuple(members), declared=declared_stacks_from_members(members),
        packed_in_scope={name: packed[name] for name in in_scope},
        omitted_outside_layer_stride=omitted)


def _project_expert_population(population: ExpertPopulation, *, weights, menus,
                               model_path, cache_dir: Path, measured=None,
                               projection=None) -> tuple[dict, dict]:
    """Ask the producer to project every in-scope stack; bind it; check the bytes.

    Each request covers the whole campaign (the producer hashes the checkpoint
    to identify its source); family retries never request one stack at a time.
    The answer is bound exactly to the
    profile-declared units -- no name outside the profile's declaration, no
    2-D slice PrismaQuant chose -- and each unit's source tensor is read from
    the shard the producer hashed and compared byte-for-byte with the live
    view this run prices.  Returns ``(carried_block, {qname: unit})``; any
    disagreement refuses by name (PrismaQuant #183).

    ``projection`` supplies a producer answer already obtained for this scope
    (the census's), so a sharded campaign asks the producer once instead of
    once per row and every shard carries the identical block.  ``measured``
    narrows only what is byte-checked and priced here; the carried block always
    covers every declared stack, because that is what the allocation rebinds.
    """
    from .tessera_expert_projection import (
        ExpertProjectionError, bind_expert_projection, carried_projection,
        producer_plan_tool, request_expert_projection, source_unit_weight,
        stack_plan_request,
    )
    from .tessera_formats import parse_tessera_format_name
    import torch

    # The producer's plan asks for a nominal rung per stack; the unit records
    # it returns do not depend on it -- the producer checks only that the
    # family has an expert route on its build.  The menu is ordered by rate,
    # so its first rung's family is arbitrary with respect to the route, and
    # asking on it refuses a stack whose OTHER families are routable.  Ask the
    # producer family by family in menu order and keep the first it accepts;
    # every refusal is carried, because "this build has no expert route for
    # this family" is a measured fact the payload should record rather than
    # lose in a traceback (PrismaQuant #280).
    if projection is not None:
        # A census already asked the producer for exactly this scope. Bind its
        # answer here anyway: binding is the check, and it is the check this
        # run needs -- the carried block is only usable if it covers every
        # declared stack with the declared geometry.
        carried = dict(projection)
        if carried.get("schema") is None or "producer" not in carried:
            raise RuntimeError(
                "the census carries no producer expert projection to reuse "
                "(PrismaQuant #183).")
        try:
            bound = bind_expert_projection(carried["producer"],
                                           declared=population.declared)
        except ExpertProjectionError as exc:
            raise RuntimeError(
                "the census's producer expert projection does not bind to this "
                f"run's declared population: {exc} (PrismaQuant #183).") from exc
        return carried, _checked_projected_units(
            bound, weights=weights, model_path=model_path,
            source=carried["producer"]["source"],
            measured=measured)

    ladders: dict[str, list[tuple[str, int]]] = {}
    for stack, units in sorted(population.declared.items()):
        first = sorted(units)[0]
        menu = list(menus.get(first) or [])
        if not menu:
            raise RuntimeError(
                f"Tessera campaign has no menu for projected expert unit {first} "
                f"(stack {stack}); refusing to price a stack the allocator could "
                "not choose a rung for (PrismaQuant #183).")
        ladder: list[tuple[str, int]] = []
        seen: set[str] = set()
        for entry in menu:
            parsed = parse_tessera_format_name(entry.format_name)
            if parsed is None:
                continue
            family, rung = parsed
            grid = family.payload_grid().name
            if grid in seen:
                continue
            seen.add(grid)
            ladder.append((grid, int(rung)))
        if not ladder:
            raise RuntimeError(
                f"Tessera campaign menu for projected expert unit {first} names no "
                "Tessera rung the producer could plan (PrismaQuant #183).")
        ladders[stack] = ladder
    out_path = Path(cache_dir) / "expert_projection.json"
    try:
        tool = producer_plan_tool()
    except ExpertProjectionError as exc:
        raise RuntimeError(
            "Tessera campaign cannot ask the producer for its expert projection; "
            f"refusing to price it: {exc} (PrismaQuant #183).") from exc
    # Keep every previously attempted mixed nominal plan. Also ask at each
    # common family by NAME: different stack menus can place E4M3 at different
    # indices, so moving all ladders together can miss the one routed family
    # even when every stack offers it (#295). At most two plans per distinct
    # family count, never the Cartesian product of stack assignments and never
    # one checkpoint-hashing producer call per stack.
    plans = [{stack: ladder[min(index, len(ladder) - 1)]
              for stack, ladder in ladders.items()}
             for index in range(max(map(len, ladders.values())))]
    by_family = {stack: dict(ladder) for stack, ladder in ladders.items()}
    common_families = set.intersection(*(set(rows) for rows in by_family.values()))
    for grid in next(iter(by_family.values())):
        if grid in common_families:
            asked = {stack: (grid, rows[grid]) for stack, rows in by_family.items()}
            if asked not in plans:
                plans.append(asked)
    attempts: list[dict] = []
    stacks: dict[str, tuple[str, int]] = {}
    answer = bound = None
    for asked in plans:
        stacks = asked
        try:
            answer = request_expert_projection(model_path, stacks, out_path=out_path)
            bound = bind_expert_projection(answer, declared=population.declared)
        except ExpertProjectionError as exc:
            attempts.append({"request": stack_plan_request(stacks), "refused": str(exc)})
            answer = bound = None
            continue
        attempts.append({"request": stack_plan_request(stacks), "refused": None})
        break
    if bound is None or answer is None:
        raise RuntimeError(
            "Tessera campaign cannot bind the producer's expert projection to the "
            "profile-declared population on the attempted nominal family plans; refusing to "
            "price it: "
            + " || ".join(a["refused"] for a in attempts)
            + " (PrismaQuant #183).")
    carried = carried_projection(answer, bound, request=stack_plan_request(stacks),
                                 tool=str(tool))
    carried["plan_attempts"] = attempts
    return carried, _checked_projected_units(
        bound, weights=weights, model_path=model_path,
        source=answer["source"], measured=measured)


def _checked_projected_units(bound, *, weights, model_path, source,
                             measured=None) -> dict[str, dict]:
    """The producer's unit records for the units this run prices, bytes checked.

    Each unit's source tensor is read from the shard the producer hashed and
    compared byte-for-byte with the live view this run prices, so the exporter
    cannot encode bytes this table did not price (PrismaQuant #183).  Only the
    ``measured`` units are read: a shard cannot check a tensor it never loaded,
    and claiming it had would be the assertion the check exists to replace.
    """
    import torch

    from .tessera_expert_projection import ExpertProjectionError, source_unit_weight

    projected: dict[str, dict] = {}
    mismatched: list[str] = []
    for _stack, units in sorted(bound.items()):
        for name, unit in sorted(units.items()):
            if measured is not None and name not in measured:
                continue
            try:
                weight = source_unit_weight(model_path, source, unit)
            except ExpertProjectionError as exc:
                raise RuntimeError(
                    f"Tessera campaign cannot read the producer's source tensor for "
                    f"{name}: {exc} (PrismaQuant #183).") from exc
            live = weights[name].detach().cpu()
            if live.dtype != weight.dtype or not torch.equal(live, weight):
                mismatched.append(
                    f"{name} (live {tuple(live.shape)} {live.dtype} vs source "
                    f"{unit['source_tensor']} {tuple(weight.shape)} {weight.dtype})")
                continue
            projected[name] = unit
    if mismatched:
        raise RuntimeError(
            "Tessera campaign's live expert view disagrees byte-for-byte with the "
            "producer's source tensor for " + ", ".join(mismatched)
            + "; the exporter would encode bytes this table did not price. "
            "Refusing (PrismaQuant #183).")
    return projected


def _population_block(*, dense_targets, expert_targets, dense_all, pinned,
                      population: ExpertPopulation, layer_stride: int,
                      costs, menus, stack_samples=None, profile=None) -> dict:
    """Distinguish selected targets from units with actual emitted prices."""
    from .tessera_expert_projection import POPULATION_SCHEMA

    dense_omitted = sorted(set(dense_all) - set(dense_targets))
    priced = {name for name, rows in costs.items() if rows}
    stack_decisions = {}
    represented = set()
    for packed_qname, sample in sorted((stack_samples or {}).items()):
        if profile is None:
            raise StackSampleError("stack population requires the model profile")
        if (packed_qname != sample.packed_qname or
                packed_qname not in population.packed_in_scope or
                int(population.packed_in_scope[packed_qname][0]) != sample.num_experts):
            raise StackSampleError(f"{packed_qname}: sample disagrees with packed population")
        roles = tuple(profile.packed_expert_projection_names(sample.packed_param))
        stem = packed_qname[: -len(sample.packed_param)].rstrip(".")
        members = {f"{stem}.{e}.{role}" for e in range(sample.num_experts) for role in roles}
        declared = set(population.declared.get(sample.packed_experts_module, {}))
        measured = {member for group in sample.members.values() for member in group}
        expected_measured = {f"{stem}.{e}.{role}" for e in sample.sampled_experts for role in roles}
        if (not members or not members <= declared or measured != expected_measured
                or not measured <= members):
            raise StackSampleError(f"{packed_qname}: sample members disagree with declared population")
        if represented & members:
            raise StackSampleError(f"{packed_qname}: source members belong to multiple stack decisions")
        if members & priced:
            raise StackSampleError(f"{packed_qname}: both packed decision and source members are priced")
        represented.update(members)
        stack_decisions[packed_qname] = {
            "stack": sample.packed_experts_module,
            "members": sorted(members), "sampled_members": sorted(measured),
        }
    # Source experts represented by an HT row are neither separate prices nor
    # unpriced BF16 units. Keep their full frame in the explicit decision map.
    expert_targets = (set(expert_targets) - represented) | set(stack_decisions)
    dense_priced = sorted(set(dense_targets) & priced)
    expert_priced = sorted(set(expert_targets) & priced)
    unpriced = {
        kind: {name: ("no_admitted_menu" if not (menus.get(name) or
                          any(menus.get(member) for member in
                              stack_decisions.get(name, {}).get("sampled_members", ())))
                      else "no_successful_anchor")
               for name in sorted(set(targets) - priced)}
        for kind, targets in (("dense", dense_targets),
                              ("routed_experts", expert_targets))
    }
    represented_priced = priced | {
        member for name, decision in stack_decisions.items() if name in priced
        for member in decision["members"]}
    complete_stacks = sorted(stack for stack, units in population.declared.items()
                             if set(units) <= represented_priced)
    packed = {name: list(shape) for name, shape
              in sorted(population.packed_in_scope.items())}
    packed_omitted = {name: list(shape) for name, shape
                      in sorted(population.omitted_outside_layer_stride.items())}
    result = {
        "schema": POPULATION_SCHEMA,
        "layer_stride": int(layer_stride),
        "enumerated": {"dense": sorted(dense_targets),
                       "routed_experts": sorted(expert_targets),
                       "packed_parameters": packed,
                       "stacks": sorted(population.declared)},
        "unpriced": unpriced,
        "priced": {
            "dense": dense_priced,
            "routed_experts": expert_priced,
            "packed_parameters": {name: shape for name, shape in packed.items()
                                  if name.rsplit(".", 1)[0] in complete_stacks},
            "stacks": complete_stacks,
        },
        "omitted": {
            "dense_outside_layer_stride": dense_omitted,
            "packed_outside_layer_stride": packed_omitted,
            "pinned": sorted(pinned),
        },
        "counts": {
            "dense_priced": len(dense_priced),
            "routed_experts_priced": len(expert_priced),
            "dense_unpriced": len(unpriced["dense"]),
            "routed_experts_unpriced": len(unpriced["routed_experts"]),
            "dense_omitted": len(dense_omitted),
            "packed_omitted": len(packed_omitted),
            "pinned": len(pinned),
        },
    }
    if stack_decisions:
        result["stack_decisions"] = stack_decisions
    return result


def campaign_population_block(**kwargs) -> dict:
    """The payload's population block, spelled for out-of-module callers.

    The merge rebuilds this block over the whole census rather than over one
    shard's selection, and it must be the same function that built the
    monolith's or the merged table would report a coverage the campaign never
    computed.
    """
    return _population_block(**kwargs)


def _format_executes_static_activation_contract(format_name: str) -> bool:
    """Does this rung's route execute a STATIC activation contract?

    The spec's answer (``FormatSpec.static_activation_contract``), which
    ``synthesize_tessera_spec`` derives from the registry row the rung's route
    names -- never a compare of that row's NAME against ``"NVFP4"`` (#205,
    #221).  Same field ``_measure_anchor`` prices through, so the rung this
    refuses to resume is exactly the rung it refuses to score.
    """
    from . import format_registry as fr

    return fr.get_format(format_name).static_activation_contract is not None


def _require_resumable_anchor(anchor: CampaignAnchor, static_scales) -> None:
    """Refuse a resumed anchor priced under a different activation contract.

    The per-row half of the resume identity rule; its one caller is
    :func:`_checkpoint_anchor_identity`, and the run-level half (this run's
    static scales and policy, bound into the journal identity) is
    :func:`_campaign_checkpoint_identity`.  Resume merges checkpoint rows into
    this run's table, and the table's rows must be one currency.  A W4A4
    anchor with no ``input_global_scale`` was measured under the pre-#194
    dynamic FP32-scale quantiser; one with a *different* scale was measured
    on a different calibration; a dynamic-route row carrying a scale was
    stamped by no producer this campaign has.  None is a price of this run's
    served A side, and merging one silently is the exact mixed-table failure
    the Hessian identity guard exists to catch on its own axis.
    """
    if not _format_executes_static_activation_contract(anchor.format_name):
        if anchor.input_global_scale is not None:
            raise ActivationScaleContractError(
                f"checkpoint anchor {anchor.qname} {anchor.format_name} "
                f"carries input_global_scale={anchor.input_global_scale!r} "
                "but its route keeps the serving format's dynamic activation "
                "quantiser: no producer of this campaign stamps a static scale "
                "on that route, so the row is not one of this run's prices.")
        return
    if anchor.input_global_scale is None:
        raise ActivationScaleContractError(
            f"checkpoint anchor {anchor.qname} {anchor.format_name} carries "
            "no input_global_scale: it was priced under the pre-served-"
            "contract dynamic activation quantiser and cannot be merged into "
            "a served-contract table. Delete the checkpoint (or pass a fresh "
            "--checkpoint) to re-measure these anchors."
        )
    expected = static_scales.get(anchor.qname)
    if expected is None or float(anchor.input_global_scale) != float(expected):
        raise ActivationScaleContractError(
            f"checkpoint anchor {anchor.qname} {anchor.format_name} was "
            f"priced at input_global_scale={anchor.input_global_scale!r} but "
            f"this run's calibration yields {expected!r}. A resumed anchor "
            "must have been scored under this run's own static scales, or "
            "the table mixes two activation calibrations under one identity."
        )


def write_export_inputs(cache_dir: Path, *, hessians, hessian_rows,
                        hessian_identity, static_scales, static_scale_policy):
    """Write the exporter's ``--hessian`` and ``--input-scales`` inputs.

    ``(hessian_capture_path | None, input_scales_path | None,
    capture_sha256 | None)``.  The allocation an allocator builds on this
    campaign's table is priced under exactly these Hessians and these static
    activation scales, so the export leg must be handed them back or the
    artifact built is not the artifact priced (RobTand/prismaquant#193).
    Both files are in the shapes Tessera's exporter consumes:

    * ``hessian_capture.pt`` -- ``{"H": {unit: XᵀX}, "counts", "provenance"}``,
      what ``ActivationSource.from_capture`` loads; the H tensors are the
      campaign's own un-normalised accumulators, the identity is the same
      triple stamped on every cost row, and ``hessian_role: "fit"`` marks it
      as bytes-shaping (Tessera refuses a held-out capture there).  The
      returned ``capture_sha256`` is the content digest of exactly this
      payload (``tessera_export_lane.hessian_capture_sha256``, Tessera's own
      seal rule); every cost row carries it and the export gate binds the
      allocation to the payload by it (RobTand/prismaquant#204).  A JSON
      sidecar ``<capture>.provenance.json`` repeats the identity and carries
      the same digest, so it can only ever describe the payload beside it:
      both files are staged and renamed, the old sidecar is removed before
      the new payload lands, and the sidecar lands last -- at no point does a
      sidecar sit beside a payload it does not seal.
    * ``input_scales.safetensors`` -- one ``<unit>.input_global_scale`` F32
      scalar per unit, the exporter's stock-NVFP4 spelling, valued exactly as
      the W4A4 costs were scored.

    ``hessians=None`` is the deliberate weights-only campaign: no capture is
    written, matching the ``supplied=false`` stamp the rows carry.
    """
    import torch

    from .tessera_export_lane import (
        HESSIAN_CAPTURE_SHA256_SCHEMA, hessian_capture_sha256,
    )

    hessian_capture_path = None
    capture_sha256 = None
    if hessians is not None:
        hessian_capture_path = cache_dir / "hessian_capture.pt"
        capture_provenance = {**dict(hessian_identity), "hessian_role": "fit"}
        saved_hessians = {name: h for name, h in hessians.items()
                          if h is not None}
        capture_sha256 = hessian_capture_sha256(saved_hessians,
                                                capture_provenance)
        sidecar = hessian_capture_path.with_name(
            hessian_capture_path.name + ".provenance.json")
        tmp_capture = hessian_capture_path.with_suffix(".pt.tmp")
        tmp_sidecar = sidecar.with_suffix(".json.tmp")
        torch.save({
            "H": saved_hessians,
            "counts": dict(hessian_rows),
            "provenance": capture_provenance,
        }, tmp_capture)
        tmp_sidecar.write_text(json.dumps({
            **capture_provenance,
            "capture_sha256": capture_sha256,
            "capture_sha256_schema": HESSIAN_CAPTURE_SHA256_SCHEMA,
        }, indent=2, sort_keys=True) + "\n")
        if sidecar.exists():
            sidecar.unlink()
        os.replace(tmp_capture, hessian_capture_path)
        os.replace(tmp_sidecar, sidecar)
        print(f"[campaign] wrote {hessian_capture_path} "
              f"({len(saved_hessians)} Hessians, capture_sha256 "
              f"{capture_sha256[:12]})", flush=True)
    input_scales_path = None
    if static_scales:
        from safetensors.torch import save_file

        from .nvfp4_activation_contract import input_global_scale_tensor

        input_scales_path = cache_dir / "input_scales.safetensors"
        save_file(
            {f"{name}.input_global_scale": input_global_scale_tensor(value)
             for name, value in static_scales.items()},
            str(input_scales_path),
            metadata={"input_global_scale_policy": str(static_scale_policy)},
        )
        print(f"[campaign] wrote {input_scales_path} "
              f"({len(static_scales)} static input scales)", flush=True)
    return hessian_capture_path, input_scales_path, capture_sha256


def main(argv: "Sequence[str] | None" = None) -> int:
    import torch

    from . import format_registry as fr
    from .production_weight_cache import ProductionWeightCache
    from .tessera_menu import MENU_MODES, PARALLEL_NONE, menu_mode
    from .tessera_rate_surface import leave_one_anchor_out
    from .tessera_render import (
        HessianContractError, tessera_encoder_hessian_status,
    )
    from .tessera_serving_scope import (
        add_serving_scope_arguments, serving_target_from_args,
        context_by_unit_from_stats, scope_provenance,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    add_serving_scope_arguments(ap)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="cost payload (.pkl)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="identity-bound JSON manifest with sibling .parts "
                         "unit shards; defaults beside --out")
    ap.add_argument("--menu-mode", default=None, choices=sorted(MENU_MODES))
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-act-rows", type=int, default=512)
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="price every Nth decoder layer (1 = every Linear)")
    ap.add_argument("--anchors", type=int, default=ROUND_ONE_ANCHORS)
    ap.add_argument("--anchor-batch-size", type=int, default=1,
                    help="maximum compatible expert anchors in one producer "
                         "batch within this action (1 = scalar). Does not "
                         "change the anchor schedule or PB placement.")
    ap.add_argument("--max-rounds", type=int, default=0,
                    help="hard stop on adaptive rounds (0 = governed by "
                         "--anchor-budget instead). Rounds are not the "
                         "budget: a round adds ONE anchor to each surface "
                         "that is still failing its gate, so capping rounds "
                         "caps how far the worst surface can be improved, "
                         "which is the opposite of what the adaptive loop is "
                         "for.")
    ap.add_argument("--rate-band", default=None,
                    help="'lo,hi' in q256 body-rate units. Round one puts a "
                         "window family's two anchors at the ends of this "
                         "band instead of at the ends of its whole realisable "
                         "range, and an audited unit gets a third inside it. "
                         "Unset reproduces every artifact built before "
                         "2026-09-06.")
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
    ap.add_argument("--units", default=None,
                    help="JSON selection of fused anchor groups to MEASURE "
                         "(prismaquant.tessera_campaign_units.v1). The scope "
                         "resolution, the census and the menus are unchanged; "
                         "only the encoding work narrows, and the checkpoint "
                         "identity narrows with it, so two invocations over "
                         "disjoint selections never contend. Omitted: measure "
                         "every group in scope, exactly as before.")
    ap.add_argument("--calibration-census", default=None,
                    help="JSON per-unit calibration row counts for the WHOLE "
                         "priced scope (prismaquant.tessera_campaign_census.v1, "
                         "written by --census-out). Every unit this run hooks "
                         "is checked against it; the run then stamps the "
                         "scope's fit_tokens rather than its own selection's, "
                         "which is what makes a sharded campaign's Hessian "
                         "identity the monolith's.")
    ap.add_argument("--seed-checkpoint", default=None,
                    help="another campaign's checkpoint manifest whose already "
                         "measured anchors this run may adopt for the units it "
                         "prices. Every adopted row goes through the SAME "
                         "per-row gates a resume does -- the producer's input "
                         "identity recomputed from this run's weights, menu, "
                         "Hessian and static scale, and the cached wire "
                         "re-verified against it -- so a row is adopted only "
                         "when its bytes are the bytes this run would have "
                         "encoded. The run-level identities need not match: "
                         "that is the point, and the difference is stamped.")
    ap.add_argument("--seed-wire-dir", default=None,
                    help="the seed checkpoint's wire cache; its blobs are "
                         "linked into this run's wire dir before verification. "
                         "Defaults to <seed cache>/wire beside the manifest.")
    ap.add_argument("--census-out", default=None,
                    help="collect the calibration census over the whole scope, "
                         "write it here and exit. No Hessians, no retained "
                         "scoring rows, no encodes.")
    args = ap.parse_args(argv)
    if args.anchor_batch_size < 1:
        ap.error("--anchor-batch-size must be positive")
    if args.anchor_batch_size > 1:
        from .tessera_render import require_tessera_batch_encoder
        require_tessera_batch_encoder()
    serving_target = serving_target_from_args(args)

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
    # The packed expert population the producer bridge covers, or a refusal by
    # name before calibration.  Its members are priced as the producer's
    # projected units below (PrismaQuant #183).
    population = _require_campaign_population(model, profile, args.layer_stride)
    dense_targets: list[str] = []
    all_dense: list[str] = []
    pinned: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.endswith("lm_head") or "embed" in name:
            continue
        if profile.is_pinned_name(name):
            pinned.append(name)
            continue
        all_dense.append(name)
    dense_targets = _campaign_layer_scope(all_dense, args.layer_stride)
    expert_targets = population.qnames
    expert_members = {member.qname: member for member in population.members}
    targets = [*dense_targets, *expert_targets]
    # The whole priced scope, before any selection narrows the encoding work.
    # It is what the census counts, what the population block enumerates, and
    # what a merge checks a set of shards covers.
    census_dense_targets = list(dense_targets)
    census_expert_targets = list(expert_targets)
    census_targets = list(targets)
    scope_groups = resolve_anchor_groups(
        census_targets, profile=profile, expert_members=expert_members)
    print(f"[campaign] {len(dense_targets)} target Linears + {len(expert_targets)} "
          f"projected expert units in {len(population.declared)} stacks, "
          f"{len(scope_groups)} anchor groups, mode={mode}, "
          f"device={device}", flush=True)

    # The census quantum: the whole prologue over the whole scope -- one
    # calibration forward that counts rows and calibrates the static A-side
    # maxima, and one producer projection request -- written out and stopped
    # before a single anchor is encoded.  Every later shard reads it, so no
    # shard re-derives a scope-wide answer and none of them can disagree
    # about one.
    census_only = bool(args.census_out)
    if census_only and args.units:
        raise RuntimeError(
            "--census-out takes the whole scope; it cannot be narrowed by --units")
    if census_only and args.calibration_census:
        raise RuntimeError("--census-out writes a census; it does not read one")

    selection = None
    stack_samples = {}
    selected_groups: list[str] = sorted(scope_groups)
    audit_units: set = set()
    inclusion_probability: dict = {}
    if args.units:
        selection = load_unit_selection(args.units)
        if (selection.get("model", args.model) != args.model
                or selection.get("layer_stride", args.layer_stride) != args.layer_stride):
            raise StackSampleError("--units model/layer_stride disagrees with this campaign")
        stack_samples = selection_stack_samples(selection, profile)
        selected_groups = select_anchor_groups(
            selection, scope_groups, where=f"--units {args.units}")
        priced, audit_units, inclusion_probability = selection_priced_units(
            selection)
        # The group's membership was already checked whole; what the sample
        # narrows is only which of those members this run encodes. Keeping the
        # two separate is what lets a sampled run stay identity-honest: the
        # checkpoint's ``units`` map holds exactly the priced units, so a
        # different draw is a different identity and cannot silently resume
        # over this one.
        keep = {name for key in selected_groups for name in scope_groups[key]
                if name in priced}
        dense_targets = [name for name in dense_targets if name in keep]
        expert_targets = [name for name in expert_targets if name in keep]
        expert_members = {name: member for name, member in expert_members.items()
                          if name in keep}
        targets = [*dense_targets, *expert_targets]
        sampled_groups = sum(1 for entry in selection["groups"]
                             if entry.get("sampled"))
        print(f"[campaign] --units selects {len(selected_groups)} of "
              f"{len(scope_groups)} anchor groups: {len(dense_targets)} dense + "
              f"{len(expert_targets)} projected expert units"
              + (f"; {sampled_groups} group(s) sampled, "
                 f"{len(audit_units)} audit unit(s)" if sampled_groups else ""),
              flush=True)
        if not targets:
            raise RuntimeError(
                f"--units {args.units}: the selection prices no unit")

    context_by_unit = None
    if serving_target is not None:
        from .sensitivity_probe import discover_moe_structure
        routed = discover_moe_structure(model, profile=profile)
        topology = {
            name: dict(zip(("router_path", "expert_id"), routed.get(name, (None, None))))
            for name in dense_targets
        }
        for name, member in expert_members.items():
            # The packed facts the probe would record for this unit: its
            # packed module and expert count, never a shape guess.
            topology[name] = {
                "_packed_experts_module": member.module_qname,
                "num_experts": int(getattr(member.module, member.param_name).shape[0]),
            }
        context_by_unit = context_by_unit_from_stats(serving_target, topology, profile)

    tokens, corpus_text = _calibration_tokens(
        args.model, args.nsamples, args.seqlen, args.seed)
    # A census run needs the counts and the maxima, which every hook produces
    # unconditionally; it needs neither the Hessians nor the retained scoring
    # rows, because it encodes nothing.
    want_h = args.hessian == "require" and not census_only
    acts, hessians, hessian_rows, act_max_abs = _collect_activations(
        model, targets, tokens, 0 if census_only else args.max_act_rows, device,
        want_hessian=want_h, profile=profile)
    # For the log line and the run-level provenance only; every encode is
    # given its own Linear's count.
    census = (None if not args.calibration_census
              else load_calibration_census(args.calibration_census, args=args))
    hessian_token_count, hessian_token_min = census_token_counts(
        census, hessian_rows)
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
    if census is not None:
        require_census_draw(census, hessian_identity,
                            where=f"--calibration-census {args.calibration_census}")

    # The static A-side calibration, from the same forward passes: one
    # input_global_scale per unit under the resolved contract policy, fused
    # siblings sharing one value.  Priced by every W4A4 anchor below and
    # written beside the payload for the export leg, so the scale that priced
    # the table is the scale the artifact serves (priced == served).
    #
    # Under a census the maxima are the SCOPE's, not this selection's, and for
    # a reason the anchor grouping does not cover: fused-sibling unification
    # (``unify_fused_sibling_max_abs``) has its own fallbacks and can group
    # units the profile's ``fused_sibling_group`` does not, so a selection that
    # is whole by the anchor partition can still be partial by the scale
    # partition -- and a partial fused group calibrates a different
    # ``input_global_scale`` than the module vLLM executes. Taking the scope's
    # maxima removes that dependence entirely, and every unit this run hooked
    # is checked against them first.
    static_scales, static_scale_policy = _static_input_scales(
        act_max_abs if census is None else census_max_abs(census, act_max_abs),
        profile=profile)

    weights = {name: dict(model.named_modules())[name].weight.detach()
               for name in dense_targets}
    for name, member in expert_members.items():
        # The profile-declared 2-D view of the live packed parameter; checked
        # byte-for-byte against the producer's source tensor below.
        weights[name] = member.weight.detach()
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
        context_by_unit=context_by_unit,
    )
    # PrismaQuant #291 (filed here first as #288). A narrowing menu mode --
    # ``attested`` without a dev pin, ``readable`` against a contract that
    # publishes no reader for these shapes -- used to resolve to nothing and
    # let the run finish successfully with ``costs: {}``. A zero-row cost
    # table is not a cheap answer, it is a missing one, and downstream it is
    # indistinguishable from a scope that legitimately holds no unit. So the
    # run refuses, and says which mode and which contract produced the
    # emptiness. This is principle 1's line: the platform reports the gap
    # instead of shipping a table nobody can tell is empty on purpose.
    no_admitted_rung = report_empty_menus(menus, mode=mode)

    # The producer's projection of every in-scope stack, asked for ONCE (it
    # hashes the whole checkpoint), bound exactly to the profile-declared
    # units, and its source bytes checked against the live views this run
    # prices.  What the producer will read at export is what is priced here.
    expert_projection = None
    projected_units: dict[str, dict] = {}
    # The projection is a SCOPE-wide answer, always: the block a shard carries
    # covers every declared stack, because the allocation rebinds the
    # producer's answer against every stack the block names
    # (``carried_units``), and a block trimmed to one shard's stacks would be
    # refused there. What narrows is only the byte-check and the priced units.
    # A shard therefore reads the census's projection rather than asking the
    # producer again -- the request hashes the whole checkpoint, so asking per
    # row would put the campaign's most expensive serial step on every row and
    # let two rows answer it differently.
    if population.declared:
        expert_projection, projected_units = _project_expert_population(
            population, weights=weights, menus=menus,
            model_path=args.model, cache_dir=cache_dir,
            measured=set(expert_targets),
            projection=(None if census is None else census.get("expert_projection")))
        print(f"[campaign] producer projected {len(expert_projection['stacks'])} stacks; "
              f"{len(projected_units)} expert units priced here", flush=True)

    if census_only:
        payload = calibration_census(
            hessian_rows, act_max_abs, args=args, groups=scope_groups,
            dense_targets=census_dense_targets,
            expert_targets=census_expert_targets,
            shapes={name: tuple(weight.shape) for name, weight in weights.items()},
            identity=hessian_identity, expert_projection=expert_projection)
        if set(payload["counts"]) != set(census_targets):
            raise RuntimeError(
                "the census did not observe every unit in scope: missing "
                + ", ".join(sorted(set(census_targets) - set(payload["counts"]))))
        out = Path(args.census_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"[campaign] wrote {out}: {len(payload['counts'])} units, rows "
              f"{min(payload['counts'].values())}..{max(payload['counts'].values())}, "
              f"{len(payload['anchor_groups'])} anchor groups", flush=True)
        return 0

    from .cost_stage_checkpoint import prepare_journal, write_unit

    # The resume identity, run level: everything a price is a function of,
    # including the static A-side contract (scales + policy) the W4A4 rows
    # are scored under.  A checkpoint from another calibration or policy is
    # refused here, by field, before a row of it is read.
    checkpoint_identity = _campaign_checkpoint_identity(
        weights=weights, acts=acts, hessians=hessians, menus=menus, args=args,
        calibration_identity=hessian_identity,
        serving_scope=(scope_provenance(serving_target, context_by_unit)
                       if serving_target is not None else None),
        static_scales=static_scales, static_scale_policy=static_scale_policy,
        expert_projection=expert_projection,
        stack_sampling_identity={name: record
            for entry in (selection or {}).get("groups", [])
            for name, record in entry.get("stack_samples", {}).items()},
    )
    journal, identity_sha256, resumed = prepare_journal(
        checkpoint.with_name(checkpoint.name + ".parts"), manifest_path=checkpoint,
        stage="Tessera campaign", resume=True, identity=checkpoint_identity,
        qnames=targets,
    )
    measured: dict[str, dict[str, list[CampaignAnchor]]] = {}
    # Rows adopted from another campaign whose rungs THIS run's menu does not
    # admit.  They are measurements of the same rate/distortion law and cost
    # the same GPU seconds to make, so they are carried rather than discarded;
    # they are held apart from ``measured`` because a price the pinned reader
    # cannot decode is not a price.  Nothing in ``costs`` is ever built from
    # this, and no fresh encode ever lands here: the menu decides what is
    # encoded, so a row can only become unservable by being adopted.
    unservable: dict[str, dict[str, dict]] = {}
    wire_records = {name: {} for name in targets}
    dirty_checkpoint_units = set()

    def adopt_state(name: str, state, *, where: str) -> None:
        """Verify one unit's stored anchors against this run and take them.

        The one path for both a resume of this run's own checkpoint and an
        adoption from another campaign's: what makes a stored row usable is
        that its inputs and its bytes are this run's, and that is checked here
        rather than inferred from which file the row came out of.

        A stored row whose rung is outside this run's menu is neither priced
        nor refused: it is recorded as ``unservable`` evidence.  The two are
        different failures.  A row that disagrees with this run's weights,
        Hessian applicability or static scale is a row about some other
        campaign and is refused by name; a row this menu does not admit is a
        correct measurement of a rung the pinned reader cannot decode, and the
        honest thing to do with it is keep it out of the prices and in the
        record.

        On the ``checkpoint`` path that branch is unreachable, and
        deliberately so: ``menu_mode`` is bound in the journal identity (it is
        one of the ``settings``, and only locations are popped), so a resume
        under a different menu is refused by ``prepare_journal`` before a row
        is read. Evidence arrives only through ``--seed-checkpoint``, which is
        the path whose whole purpose is to cross that boundary.
        """
        if not isinstance(state, dict) \
                or set(state) - {"unservable"} != {"anchors", "wire_records"} \
                or not isinstance(state["anchors"], list) \
                or not isinstance(state["wire_records"], dict):
            raise RuntimeError(f"{where} state has an invalid anchor/record envelope for {name}")
        if name not in menus:
            raise RuntimeError(f"{where} state names a unit this run does not price: {name}")
        on_menu = {entry.format_name for entry in menus[name]}
        formats = set()
        for row in state["anchors"]:
            anchor = CampaignAnchor(**row)
            if anchor.qname != name or anchor.format_name in formats:
                raise RuntimeError(f"{where} anchor has a wrong or duplicate unit/rung: {name}")
            formats.add(anchor.format_name)
            if anchor.format_name not in state["wire_records"]:
                raise RuntimeError(f"{where} anchor has no priced-wire receipt: {name}")
            if anchor.format_name not in on_menu:
                unservable.setdefault(name, {})[anchor.format_name] = {
                    "anchor": dict(row),
                    "wire_record": dict(state["wire_records"][anchor.format_name]),
                    "adopted_from": where,
                }
                continue
            # Row level, the same rule: the row's inputs (Hessian
            # applicability, static scale) must be what this run's producer
            # stamps for its rung, then its wire receipt must verify.
            identity = _checkpoint_anchor_identity(
                anchor, weights=weights, menus=menus,
                calibration_source=calibration_source, static_scales=static_scales,
                projected_units=projected_units)
            wire_records[name][anchor.format_name] = _checkpoint_wire_record(
                anchor, wire_dir, identity, existing=state["wire_records"][anchor.format_name])
            measured.setdefault(name, {}).setdefault(anchor.family, []).append(anchor)
        if formats != set(state["wire_records"]):
            raise RuntimeError(f"{where} has wire receipts outside its measured anchors: {name}")
        # Evidence the source had already set aside stays set aside, unless
        # this run's menu admits the rung -- in which case the source and this
        # run disagree about what is servable, and a merge of the two would
        # hold the same row on both sides of the line.
        for fmt, record in (state.get("unservable") or {}).items():
            if fmt in on_menu:
                raise RuntimeError(
                    f"{where} carries {fmt} for {name} as unservable and this "
                    "run's menu admits that rung")
            unservable.setdefault(name, {}).setdefault(fmt, record)

    for name, state in resumed.items():
        adopt_state(name, state, where="checkpoint")
    if resumed:
        print(f"[campaign] resumed {sum(len(v) for f in measured.values() for v in f.values())} "
              f"verified anchors from {checkpoint}", flush=True)

    seed_provenance = None
    if args.seed_checkpoint:
        seed_provenance = _adopt_seed_checkpoint(
            args.seed_checkpoint, args.seed_wire_dir,
            targets=[name for name in targets if name not in resumed],
            wire_dir=wire_dir, adopt=adopt_state,
            admits=lambda name, fmt: any(
                entry.format_name == fmt for entry in menus.get(name, ())),
            identity_sha256=identity_sha256)
        for name in seed_provenance["units"]:
            dirty_checkpoint_units.add(name)

    # The export leg's inputs, written AFTER the resume identity has accepted
    # this run's inputs and BEFORE the anchor loop.  Before the loop, so even
    # a deadline-stopped campaign leaves them (RobTand/prismaquant#193); after
    # the gate, because they are the export half of whatever table survives a
    # refusal (RobTand/prismaquant#211).  A refused resume leaves the
    # checkpoint and the previous cost file alone, so it must leave these
    # alone too -- overwriting them with the refused draw's Hessians and
    # scales strands the surviving table, which can then only be re-priced
    # from scratch.  Nothing above consumes the returned paths or digest, and
    # an accepted resume re-writes byte-identical files: the identity binds
    # ``hessians``, ``static_scales`` and ``static_scale_policy``, which are
    # exactly this call's inputs.
    # The capture's ``counts`` describe the DRAW over the priced scope, so a
    # shard writes the census's counts rather than its selection's: the merged
    # capture is then the whole scope's H under the whole scope's counts --
    # exactly the object a whole-scope run writes -- and the merge can prove it
    # by recomputing the digest.
    hessian_capture_path, input_scales_path, capture_sha256 = write_export_inputs(
        cache_dir,
        hessians=hessians if want_h else None,
        hessian_rows=(hessian_rows if census is None else census["counts"]),
        hessian_identity=hessian_identity,
        static_scales=static_scales,
        static_scale_policy=static_scale_policy,
    )

    # PrismaQuant #291 (filed here first as #288). A narrowing menu mode --
    # ``attested`` without a dev pin, ``readable`` against a contract that
    # publishes no reader for these shapes -- used to resolve to nothing and
    # let the run finish successfully with ``costs: {}``. A zero-row cost
    # table is not a cheap answer, it is a missing one, and downstream it is
    # indistinguishable from a scope that legitimately holds no unit. So the
    # run refuses, and says which mode and which contract produced it.
    #
    # Two things about where this sits. It is AFTER the journal gates because
    # an empty menu is the weakest diagnosis this run can offer: if the
    # checkpoint it was pointed at also describes different weights, a
    # different calibration or a different menu, THAT is what the operator has
    # to fix, and refusing the other way round would hide it behind "no rung
    # admitted". And it is after ``write_export_inputs`` because the Hessian
    # capture and the static A-side scales are facts about the calibration,
    # not about the menu: they were measured correctly, the next run under a
    # menu that admits something will want them, and throwing them away would
    # charge a second forward for a flag change. What is refused is the empty
    # cost table, and only that -- nothing has been encoded here either way.
    if menus and len(no_admitted_rung) == len(menus):
        print(f"[campaign] mode={mode} admits no rung for any of "
              f"{len(menus)} units against {contract_source_label()}; "
              "refusing to write an empty cost table", file=sys.stderr,
              flush=True)
        return EXIT_EMPTY_MENU

    def flush_checkpoint() -> None:
        for name in sorted(dirty_checkpoint_units):
            rows = [vars(anchor) for anchors in measured.get(name, {}).values()
                    for anchor in anchors]
            state = {"anchors": rows, "wire_records": wire_records[name]}
            if unservable.get(name):
                state["unservable"] = unservable[name]
            write_unit(journal, stage="Tessera campaign", qname=name,
                       identity_sha256=identity_sha256, state=state)
        dirty_checkpoint_units.clear()

    # Adopted rows are journalled BEFORE the anchor loop, because the loop can
    # end without reaching its own flush: a group whose gate is already closed
    # has nothing pending in round one and breaks out.  A seeded run is exactly
    # that case, so without this the units a seed adopted would price into
    # ``cost.pkl`` and leave no journal shard for ``merge`` to carry.
    if dirty_checkpoint_units:
        flush_checkpoint()

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
        return anchor_group_key(name, profile=profile, expert_members=expert_members)

    anchor_groups = resolve_anchor_groups(
        targets, profile=profile, expert_members=expert_members)

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
    rate_band = parse_rate_band(getattr(args, "rate_band", None))
    if rate_band is not None:
        print(f"[campaign] rate band q256 {rate_band[0]}..{rate_band[1]}: two "
              "anchors per window family at the band ends"
              + (f", a third for {len(audit_units)} audit unit(s)"
                 if audit_units else ""), flush=True)
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
                    want = round_one_rates(allowed, band=rate_band,
                                           anchors=args.anchors, snap=_snap)
                    extra = (None if not audit_units else
                             audit_extra_rate(allowed, want, snap=_snap))
                    for m in members:
                        rates = set(want)
                        if extra is not None and m in audit_units:
                            rates.add(extra)
                        have = {a.body_rate_q256
                                for a in measured.get(m, {}).get(family, [])}
                        pending.extend((m, family, rate)
                                       for rate in sorted(rates - have))
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
                    # LOO cannot judge two endpoints. Bootstrap an interior
                    # anchor through next_anchor_rate's widest-gap fallback;
                    # missing LOO is neither a closed gate nor a refusal.
                    worst_loo = {}
                elif not interpolable:
                    surface_stop.setdefault((key, family), "non_interpolable")
                    continue
                elif worst <= args.loo_gate:
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
        batches = _anchor_batches(
            [item for item in pending if acts.get(item[0]) is not None],
            weights=weights, expert_members=expert_members,
            batch_size=args.anchor_batch_size)
        completed = 0
        for batch in batches:
            if out_of_time():
                stopped_early = True
                print("[campaign] deadline reached; stopping", flush=True)
                break
            names = [item[0] for item in batch]
            family, rung = batch[0][1:]
            fmt = f"{family}_R{rung}"
            try:
                common = dict(format_name=fmt, cache=cache, wire_dir=wire_dir,
                    activation_kwargs_for=(
                        _activation_kwargs_for if want_h else None),
                    hessian_required=want_h)
                if len(batch) == 1:
                    name = names[0]
                    anchors = [_measure_anchor(
                        qname=name, weight=weights[name].to(device),
                        activations=acts[name].to(device),
                        static_input_scale=static_scales.get(name), **common)]
                else:
                    anchors = _measure_anchor_batch(
                        qnames=names,
                        weights=[weights[name].to(device) for name in names],
                        activations=[acts[name].to(device) for name in names],
                        static_input_scales=static_scales, **common)
            except (HessianContractError, ActivationScaleContractError):
                raise
            except Exception as exc:
                print(f"[campaign] {names} {fmt}: FAILED {type(exc).__name__}: "
                      f"{exc}", flush=True)
                continue
            for anchor in anchors:
                name = anchor.qname
                identity = _checkpoint_anchor_identity(
                    anchor, weights=weights, menus=menus,
                    calibration_source=calibration_source, static_scales=static_scales,
                    projected_units=projected_units)
                wire_records[name][fmt] = _checkpoint_wire_record(anchor, wire_dir, identity)
                measured.setdefault(name, {}).setdefault(family, []).append(anchor)
                dirty_checkpoint_units.add(name)
            completed += len(anchors)
            # Commit every joined quantum before advancing. The scalar mode
            # keeps its existing ten-anchor flush cadence.
            if args.anchor_batch_size > 1 or (completed - 1) % 10 == 0:
                flush_checkpoint()
                print(f"[campaign] r{round_index} {completed}/{len(pending)} "
                      f"batch={len(anchors)} {fmt} "
                      f"encode_seconds={sum(a.seconds for a in anchors):.3f}", flush=True)
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
            # Units the mode admitted no rung for. Empty on a healthy run;
            # never absent, so a reader never has to guess whether the run
            # was asked the question.
            "no_admitted_rung": list(no_admitted_rung),
            # Where round one put the anchors, and (when the planner sampled)
            # which experts stand for their stack and under what inclusion
            # probability. This travels with the prices because an estimate
            # built from them is only unbiased if the reader knows the pi it
            # was drawn under; the packed draw records build the stack rows.
            "rate_band": (None if rate_band is None
                          else [int(rate_band[0]), int(rate_band[1])]),
            "unit_selection_sample": {
                "audit_units": sorted(audit_units),
                "inclusion_probability": {
                    name: float(inclusion_probability[name])
                    for name in sorted(inclusion_probability)},
            },
            "tp_degree": int(args.tp_degree),
            "model": str(args.model),
            "nsamples": int(args.nsamples),
            "seqlen": int(args.seqlen),
            "max_act_rows": int(args.max_act_rows),
            "layer_stride": int(args.layer_stride),
            **({PROJECTION_KEY: expert_projection}
               if expert_projection is not None else {}),
            "anchors_round_one": int(args.anchors),
            "max_rounds": int(args.max_rounds),
            "anchor_budget": int(args.anchor_budget),
            "rounds_run": int(round_index),
            "anchor_placement": "fused_group",
            "anchor_groups": {
                key: list(members)
                for key, members in sorted(anchor_groups.items())
            },
            # What this invocation MEASURED, out of what the scope resolves to.
            # A merge reads this to prove a set of shards covers the scope
            # exactly once; a whole-scope run selects every group and says so.
            # Rows this run did not encode itself, and where they came from.
            # None when every row was measured here.
            "seed_checkpoint": seed_provenance,
            "unit_selection": ({**selection, "selected": True} if selection else {
                "schema": UNITS_SCHEMA,
                "selected": True if args.units else False,
                "groups": [
                    {"key": key, "members": list(scope_groups[key])}
                    for key in sorted(selected_groups)
                ],
            }),
            # The scope every shard shares: the enumeration the population
            # block is built from, the full grouping, and the census the
            # Hessian identity's token counts came from.
            "campaign_scope": {
                "dense_targets": sorted(census_dense_targets),
                "expert_targets": sorted(census_expert_targets),
                "dense_all": sorted(all_dense),
                "pinned": sorted(pinned),
                "declared_stacks": {
                    stack: {name: list(shape) for name, shape in sorted(units.items())}
                    for stack, units in sorted(population.declared.items())},
                "packed_in_scope": {
                    name: list(shape) for name, shape
                    in sorted(population.packed_in_scope.items())},
                "packed_outside_layer_stride": {
                    name: list(shape) for name, shape
                    in sorted(population.omitted_outside_layer_stride.items())},
                "anchor_groups": {
                    key: list(members) for key, members in sorted(scope_groups.items())},
                "calibration_census": (
                    None if census is None else
                    {"counts": dict(census["counts"]),
                     "token_count": int(hessian_token_count),
                     "token_count_min": int(hessian_token_min)}),
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
            # Adopted rows this run's menu does not admit: the rate/distortion
            # law keeps them, the allocator never sees them.  Always present,
            # so "is there evidence outside the priced menu" is a lookup and
            # not a distinction between an absent key and an empty one.
            "unservable": {
                name: {fmt: rows[fmt] for fmt in sorted(rows)}
                for name, rows in sorted(unservable.items())
            },
            "loo_gate": float(args.loo_gate),
            "max_artifact_bpp": float(args.max_artifact_bpp),
            "stopped_early": bool(stopped_early),
            "wall_seconds": time.time() - started,
            "cache_dir": str(cache_dir),
            **({"tessera_serving_scope": scope_provenance(serving_target, context_by_unit)}
               if serving_target is not None else {}),
            "wire_dir": str(wire_dir),
            "tessera_commit": os.environ.get("TESSERA_COMMIT", ""),
            # The static A-side identity every W4A4 row was priced under: the
            # policy names the formula, the values are the F32-rounded scalars
            # an exported input_global_scale tensor carries, fused siblings
            # unified. The served contract reads exactly one such scalar per
            # module (trellis_input_global_scale), so this block is what makes
            # "the priced A side is the served A side" checkable downstream.
            "activation_static_scales": {
                "policy": str(static_scale_policy),
                "source": "campaign_calibration_amax_fused_unified",
                "path": (None if input_scales_path is None
                         else str(input_scales_path)),
                "units": {name: float(value)
                          for name, value in sorted(static_scales.items())},
            },
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
                # The exporter-shaped capture of the exact Hessians above,
                # for the export leg's --hessian input; None on --hessian off.
                "capture_path": (None if hessian_capture_path is None
                                 else str(hessian_capture_path)),
                # The content digest of that payload (Tessera's own seal
                # rule), stamped on every row so the allocation binds to
                # the capture BY CONTENT, not by the draw's triple alone
                # (RobTand/prismaquant#204); None on --hessian off.
                "capture_sha256": capture_sha256,
                "token_count": int(hessian_token_count),
                "token_count_min": int(hessian_token_min),
                # The calibration identity verbatim, under its own key, so a
                # merge can hand exactly this dict back to
                # ``write_export_inputs`` instead of reconstructing it by
                # subtracting the keys around it.
                "calibration_identity": dict(hessian_identity),
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
        measured, menus, loo=loo, provenance=provenance,
        wire_backed=frozenset(projected_units), stack_samples=stack_samples)
    # Empty menus, failed anchors and interrupted work do not establish a
    # price. Publish coverage only after the cost rows have been constructed.
    payload["provenance"][POPULATION_KEY] = _population_block(
        dense_targets=dense_targets, expert_targets=expert_targets,
        dense_all=all_dense, pinned=pinned, population=population,
        layer_stride=int(args.layer_stride), costs=payload["costs"], menus=menus,
        stack_samples=stack_samples, profile=profile)
    if projected_units:
        # The producer's receipts for every priced expert wire, keyed by unit
        # then rung; the allocator carries the selected rung's receipt into
        # the allocation and the export lane hands the bytes to the exporter
        # unchanged (PrismaQuant #183).
        payload[EXPERT_WIRES_KEY] = {
            name: {fmt: dict(record) for fmt, record in sorted(wire_records[name].items())}
            for name in sorted(projected_units) if wire_records.get(name)
        }
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
