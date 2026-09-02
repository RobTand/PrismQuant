"""Price Tessera's continuous rate axis by anchor campaign.

Three families address ~3000 rungs at every shape, and one encode is seconds.
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
import json
import math
import os
import pickle
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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

def _encode_and_render(weight, format_name: str):
    """``(render, blob)`` for one rung: the bytes, and what they decode to.

    Goes through ``encode_linear`` rather than ``render_tessera_weight`` so the
    cache can hold the wire, and the render is ``read_unit_artifact(blob)`` --
    **the bytes, decoded**.  So the tensor this campaign prices is the artifact
    the export leg will ship, by construction rather than by agreement between
    two code paths (principle 8).

    ``verify=False`` for the same reason, not to save the decode: ``verify``
    compares ``read_unit_artifact`` against the encoder's in-memory
    ``reconstruct_unit``, and this function never uses the latter, so the
    comparison would be checking a tensor nothing downstream sees.  The
    encoder's own invariant is still pinned, once, by
    ``tests/test_tessera_campaign.py``.
    """
    from tessera.export import encode_linear
    from tessera.unit_artifact import read_unit_artifact

    from .tessera_formats import parse_tessera_format_name
    from .tessera_render import _grid_for

    parsed = parse_tessera_format_name(format_name)
    if parsed is None:
        raise ValueError(f"{format_name!r} is not a Tessera format name")
    family, rung = parsed
    grid = _grid_for(family)
    unit = encode_linear(
        weight, grid=grid, q256=int(rung), name=format_name, verify=False,
    )
    render = read_unit_artifact(unit.blob, device=str(weight.device))
    return render.to(dtype=weight.dtype, device=weight.device), unit.blob


def _measure_anchor(
    *, qname: str, weight, activations, format_name: str, cache, wire_dir: Path,
):
    """Render one rung, price it as served, and store the wire beside it."""
    import torch

    from . import format_registry as fr
    from .production_weight_cache import (
        _local_forward_render_score, _store_rendered_weight_entry,
    )
    from .tessera_formats import parse_tessera_format_name

    spec = fr.get_format(format_name)
    family, rung = parse_tessera_format_name(format_name)
    started = time.time()
    render, blob = _encode_and_render(weight, format_name)
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

def _collect_activations(model, targets, tokens, max_rows: int, device):
    """One forward pass per calibration batch, keeping input rows per Linear."""
    import torch

    store: dict[str, list] = {name: [] for name in targets}
    kept: dict[str, int] = {name: 0 for name in targets}
    handles = []

    def make_hook(name):
        def hook(_module, args):
            if kept[name] >= max_rows or not args:
                return
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            flat = x.detach().reshape(-1, x.shape[-1])
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
    return {
        name: (torch.cat(chunks, dim=0) if chunks else None)
        for name, chunks in store.items()
    }


def _calibration_tokens(model_path: str, n: int, seqlen: int, seed: int):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(data["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    generator = torch.Generator().manual_seed(int(seed))
    out = []
    for _ in range(int(n)):
        start = int(torch.randint(
            0, max(1, ids.shape[0] - seqlen - 1), (1,), generator=generator,
        ).item())
        out.append(ids[start:start + seqlen].unsqueeze(0))
    return out


def main(argv: "Sequence[str] | None" = None) -> int:
    import torch

    from . import format_registry as fr
    from .production_weight_cache import ProductionWeightCache
    from .tessera_menu import (
        MENU_MODES, PARALLEL_NONE, expand_tessera_menu, menu_mode,
    )
    from .tessera_rate_surface import leave_one_anchor_out

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
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--loo-gate", type=float, default=0.25,
                    help="max |log2 error| an interpolated surface may carry")
    ap.add_argument("--tp-degree", type=int, default=1)
    ap.add_argument("--max-artifact-bpp", type=float, default=0.0,
                    help="cap the MEASURED envelope at this artifact bpp "
                         "(0 = each family's whole legal range). A budget "
                         "decision, not a menu one: rungs above the envelope "
                         "stay in the menu and are simply not priced, because "
                         "the surface refuses to extrapolate.")
    ap.add_argument("--deadline-seconds", type=float, default=0.0,
                    help="stop starting new anchors after this much wall time")
    args = ap.parse_args(argv)

    mode = menu_mode(args.menu_mode)
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

    tokens = _calibration_tokens(
        args.model, args.nsamples, args.seqlen, args.seed)
    acts = _collect_activations(
        model, targets, tokens, args.max_act_rows, device)
    print("[campaign] activations collected", flush=True)

    weights = {name: dict(model.named_modules())[name].weight.detach()
               for name in targets}
    del model
    torch.cuda.empty_cache()

    cache = ProductionWeightCache(
        weights={}, levers={"tessera_campaign": True},
        cache_dir=str(cache_dir),
        metadata={"schema": SCHEMA, "menu_mode": mode},
    )
    menus: dict[str, list] = {}
    for name in targets:
        shape = tuple(weights[name].shape)
        menus[name] = expand_tessera_menu(
            shape, mode=mode, tp_degree=args.tp_degree,
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
    families_by_unit: dict[str, dict[str, tuple[int, int]]] = {}
    for name in targets:
        bounds: dict[str, tuple[int, int]] = {}
        for rung in menus[name]:
            if cap > 0 and rung.bpp > cap:
                continue
            lo, hi = bounds.get(rung.family, (rung.body_rate_q256,) * 2)
            bounds[rung.family] = (
                min(lo, rung.body_rate_q256), max(hi, rung.body_rate_q256))
        families_by_unit[name] = bounds

    # Round 1, breadth-first over units, so a deadline yields every unit priced
    # at the same depth rather than a prefix priced deeply and a tail not at all.
    for round_index in range(1, int(args.max_rounds) + 1):
        pending: list[tuple[str, str, int]] = []
        for name in targets:
            for family, (lo, hi) in families_by_unit[name].items():
                have = sorted(
                    a.body_rate_q256
                    for a in measured.get(name, {}).get(family, [])
                )
                if round_index == 1:
                    want = anchor_schedule(lo, hi, args.anchors)
                    pending.extend(
                        (name, family, r) for r in want if r not in have)
                    continue
                if len(have) < 3:
                    continue
                surface_loo = _loo_for(measured[name][family], leave_one_anchor_out)
                worst = float(surface_loo.get("max_abs_log2_error", 0.0) or 0.0)
                if worst <= args.loo_gate:
                    continue
                nxt = next_anchor_rate(have, surface_loo)
                if nxt is not None:
                    pending.append((name, family, nxt))
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
                )
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
            "loo_gate": float(args.loo_gate),
            "max_artifact_bpp": float(args.max_artifact_bpp),
            "stopped_early": bool(stopped_early),
            "wall_seconds": time.time() - started,
            "cache_dir": str(cache_dir),
            "wire_dir": str(wire_dir),
            "tessera_commit": os.environ.get("TESSERA_COMMIT", ""),
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
