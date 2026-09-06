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
``read_unit_artifact(encode_linear(...).blob)`` -- **the bytes, decoded**.  So the
cache entry holds the wire beside the dequantised render. Checkpoint resume
verifies those bytes against their producer input receipt. That proves the
cached wire is the priced wire; export reuse and served qualification still
owe their own receipts. The packed producer-plan/cached-wire bridge remains
the separate work tracked by PrismaQuant #183 (principle 8).

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
    "SCHEMA",
    "CampaignAnchor",
    "anchor_schedule",
    "campaign_cost_payload",
    "main",
    "next_anchor_rate",
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
    import torch

    from . import format_registry as fr
    from .production_weight_cache import (
        _local_forward_render_score, _store_rendered_weight_entry,
    )
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
        activation_quantize=activation_qdq,
        # No pre-clip on top: the static route's clamp lives inside the served
        # oracle (``nvfp4_activation_qdq_served`` clamps at the stored UE4M3
        # scale), and the dynamic routes have no calibrated clip at all.
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
    wire_backed: "frozenset[str] | set[str]" = frozenset(),
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
                                  expert_projection=None):
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
    # Locations and a wall-clock interruption limit are not encoding/scoring
    # inputs. All other explicit campaign settings remain bound by default.
    for name in ("out", "cache_dir", "checkpoint", "deadline_seconds"):
        settings.pop(name, None)
    return {
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


def _collect_activations(model, targets, tokens, max_rows: int, device,
                         *, want_hessian: bool = False, profile=None):
    """One model forward per batch, for dense and declared packed projections.

    Returns ``(rows, hessians, token_counts, max_abs)``.

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
    This capture API does not open the main-entry packed-population/export
    gate: :func:`_require_campaign_population` still refuses a live packed
    population before calibration, so nothing here reaches a cost payload or
    the export inputs until the producer's projection bridge exists
    (PrismaQuant #183).
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
    rows = {
        name: (torch.cat(chunks, dim=0) if chunks else None)
        for name, chunks in store.items()
    }
    hessians = {
        name: (None if h is None else h.to(device="cpu"))
        for name, h in hess.items()
    } if want_hessian else {}
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
                               model_path, cache_dir: Path) -> tuple[dict, dict]:
    """Ask the producer to project every in-scope stack; bind it; check the bytes.

    One subprocess for the whole campaign (the producer hashes the checkpoint
    to identify its source).  The answer is bound exactly to the
    profile-declared units -- no name outside the profile's declaration, no
    2-D slice PrismaQuant chose -- and each unit's source tensor is read from
    the shard the producer hashed and compared byte-for-byte with the live
    view this run prices.  Returns ``(carried_block, {qname: unit})``; any
    disagreement refuses by name (PrismaQuant #183).
    """
    from .tessera_expert_projection import (
        ExpertProjectionError, bind_expert_projection, carried_projection,
        producer_plan_tool, request_expert_projection, source_unit_weight,
        stack_plan_request,
    )
    from .tessera_formats import parse_tessera_format_name
    import torch

    # The producer's plan asks for a nominal rung per stack; the unit records
    # it returns do not depend on it.  The first menu rung of the stack's
    # first member is the nominal one (every member has the same menu shape
    # class, and the allocator picks the served rung later).
    stacks: dict[str, tuple[str, int]] = {}
    for stack, units in sorted(population.declared.items()):
        first = sorted(units)[0]
        menu = list(menus.get(first) or [])
        if not menu:
            raise RuntimeError(
                f"Tessera campaign has no menu for projected expert unit {first} "
                f"(stack {stack}); refusing to price a stack the allocator could "
                "not choose a rung for (PrismaQuant #183).")
        parsed = parse_tessera_format_name(menu[0].format_name)
        if parsed is None:
            raise RuntimeError(
                f"Tessera campaign menu for projected expert unit {first} opens with a "
                f"non-Tessera rung {menu[0].format_name!r}; the producer cannot plan it "
                "(PrismaQuant #183).")
        family, rung = parsed
        stacks[stack] = (family.payload_grid().name, int(rung))
    out_path = Path(cache_dir) / "expert_projection.json"
    try:
        tool = producer_plan_tool()
        projection = request_expert_projection(model_path, stacks, out_path=out_path)
        bound = bind_expert_projection(projection, declared=population.declared)
    except ExpertProjectionError as exc:
        raise RuntimeError(
            "Tessera campaign cannot bind the producer's expert projection to the "
            f"profile-declared population; refusing to price it: {exc} (PrismaQuant #183)."
        ) from exc
    carried = carried_projection(projection, bound, request=stack_plan_request(stacks),
                                 tool=str(tool))
    projected: dict[str, dict] = {}
    mismatched: list[str] = []
    for stack, units in sorted(bound.items()):
        for name, unit in sorted(units.items()):
            try:
                source = source_unit_weight(model_path, projection["source"], unit)
            except ExpertProjectionError as exc:
                raise RuntimeError(
                    f"Tessera campaign cannot read the producer's source tensor for "
                    f"{name}: {exc} (PrismaQuant #183).") from exc
            live = weights[name].detach().cpu()
            if live.dtype != source.dtype or not torch.equal(live, source):
                mismatched.append(
                    f"{name} (live {tuple(live.shape)} {live.dtype} vs source "
                    f"{unit['source_tensor']} {tuple(source.shape)} {source.dtype})")
                continue
            projected[name] = unit
    if mismatched:
        raise RuntimeError(
            "Tessera campaign's live expert view disagrees byte-for-byte with the "
            "producer's source tensor for " + ", ".join(mismatched)
            + "; the exporter would encode bytes this table did not price. "
            "Refusing (PrismaQuant #183).")
    return carried, projected


def _population_block(*, dense_targets, expert_targets, dense_all, pinned,
                      population: ExpertPopulation, layer_stride: int,
                      costs, menus) -> dict:
    """Distinguish selected targets from units with actual emitted prices."""
    from .tessera_expert_projection import POPULATION_SCHEMA

    dense_omitted = sorted(set(dense_all) - set(dense_targets))
    priced = {name for name, rows in costs.items() if rows}
    dense_priced = sorted(set(dense_targets) & priced)
    expert_priced = sorted(set(expert_targets) & priced)
    unpriced = {
        kind: {name: ("no_admitted_menu" if not menus.get(name)
                      else "no_successful_anchor")
               for name in sorted(set(targets) - priced)}
        for kind, targets in (("dense", dense_targets),
                              ("routed_experts", expert_targets))
    }
    complete_stacks = sorted(stack for stack, units in population.declared.items()
                             if set(units) <= priced)
    packed = {name: list(shape) for name, shape
              in sorted(population.packed_in_scope.items())}
    packed_omitted = {name: list(shape) for name, shape
                      in sorted(population.omitted_outside_layer_stride.items())}
    return {
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
    print(f"[campaign] {len(dense_targets)} target Linears + {len(expert_targets)} "
          f"projected expert units in {len(population.declared)} stacks, mode={mode}, "
          f"device={device}", flush=True)

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
    want_h = args.hessian == "require"
    acts, hessians, hessian_rows, act_max_abs = _collect_activations(
        model, targets, tokens, args.max_act_rows, device,
        want_hessian=want_h, profile=profile)
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

    # The static A-side calibration, from the same forward passes: one
    # input_global_scale per unit under the resolved contract policy, fused
    # siblings sharing one value.  Priced by every W4A4 anchor below and
    # written beside the payload for the export leg, so the scale that priced
    # the table is the scale the artifact serves (priced == served).
    static_scales, static_scale_policy = _static_input_scales(
        act_max_abs, profile=profile)

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

    # The producer's projection of every in-scope stack, asked for ONCE (it
    # hashes the whole checkpoint), bound exactly to the profile-declared
    # units, and its source bytes checked against the live views this run
    # prices.  What the producer will read at export is what is priced here.
    expert_projection = None
    projected_units: dict[str, dict] = {}
    if population.declared:
        expert_projection, projected_units = _project_expert_population(
            population, weights=weights, menus=menus, model_path=args.model,
            cache_dir=cache_dir)
        print(f"[campaign] producer projected {len(projected_units)} expert units in "
              f"{len(expert_projection['stacks'])} stacks", flush=True)

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
    )
    journal, identity_sha256, resumed = prepare_journal(
        checkpoint.with_name(checkpoint.name + ".parts"), manifest_path=checkpoint,
        stage="Tessera campaign", resume=True, identity=checkpoint_identity,
        qnames=targets,
    )
    measured: dict[str, dict[str, list[CampaignAnchor]]] = {}
    wire_records = {name: {} for name in targets}
    dirty_checkpoint_units = set()
    for name, state in resumed.items():
        if set(state) != {"anchors", "wire_records"} or not isinstance(state["anchors"], list) \
                or not isinstance(state["wire_records"], dict):
            raise RuntimeError(f"checkpoint state has an invalid anchor/record envelope for {name}")
        formats = set()
        for row in state["anchors"]:
            anchor = CampaignAnchor(**row)
            if anchor.qname != name or anchor.format_name in formats:
                raise RuntimeError(f"checkpoint anchor has a wrong or duplicate unit/rung: {name}")
            formats.add(anchor.format_name)
            if anchor.format_name not in state["wire_records"]:
                raise RuntimeError(f"checkpoint anchor has no priced-wire receipt: {name}")
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
            raise RuntimeError(f"checkpoint has wire receipts outside its measured anchors: {name}")
    if resumed:
        print(f"[campaign] resumed {sum(len(v) for f in measured.values() for v in f.values())} "
              f"verified anchors from {checkpoint}", flush=True)

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
    hessian_capture_path, input_scales_path, capture_sha256 = write_export_inputs(
        cache_dir,
        hessians=hessians if want_h else None,
        hessian_rows=hessian_rows,
        hessian_identity=hessian_identity,
        static_scales=static_scales,
        static_scale_policy=static_scale_policy,
    )

    def flush_checkpoint() -> None:
        for name in sorted(dirty_checkpoint_units):
            rows = [vars(anchor) for anchors in measured.get(name, {}).values()
                    for anchor in anchors]
            write_unit(journal, stage="Tessera campaign", qname=name,
                       identity_sha256=identity_sha256,
                       state={"anchors": rows, "wire_records": wire_records[name]})
        dirty_checkpoint_units.clear()

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
        # A projected expert unit anchors with its whole stack: the producer
        # plans ONE rung per stack, so every member must measure the same
        # rungs for the allocator's stack-uniform choice to have a priced
        # (and wire-backed) row on every member.
        member = expert_members.get(name)
        if member is not None:
            return f"s:{member.module_qname}"
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
                    static_input_scale=static_scales.get(name),
                )
            except (HessianContractError, ActivationScaleContractError):
                # Never absorbed into "one anchor failed": a contract refusal
                # is about every row this run would write, not this one.
                raise
            except Exception as exc:
                print(f"[campaign] {name} {fmt}: FAILED {type(exc).__name__}: "
                      f"{exc}", flush=True)
                continue
            identity = _checkpoint_anchor_identity(
                anchor, weights=weights, menus=menus,
                calibration_source=calibration_source, static_scales=static_scales,
                projected_units=projected_units)
            wire_records[name][fmt] = _checkpoint_wire_record(anchor, wire_dir, identity)
            measured.setdefault(name, {}).setdefault(family, []).append(anchor)
            dirty_checkpoint_units.add(name)
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
        wire_backed=frozenset(projected_units))
    # Empty menus, failed anchors and interrupted work do not establish a
    # price. Publish coverage only after the cost rows have been constructed.
    payload["provenance"][POPULATION_KEY] = _population_block(
        dense_targets=dense_targets, expert_targets=expert_targets,
        dense_all=all_dense, pinned=pinned, population=population,
        layer_stride=int(args.layer_stride), costs=payload["costs"], menus=menus)
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
