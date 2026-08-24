"""Exact two-host transient AURA campaign for the strict RTX 4090 lane.

The ordinary format-menu production-cache path retains rendered weights.  This
driver instead partitions *qnames* by whole decoder layer, transiently renders
three lattice anchors plus fresh native FP8 through the existing streamed AURA
consumer, and merges the two receipt-bearing scalar shards before anchored
interpolation and one byte-budget solve.
Nothing in this module publishes or validates an exported artifact.
"""
from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any

from prismaquant.anchored_cost import (
    RenderRequest,
    candidates_by_segment,
    price_anchored_candidates,
    run_allocator_once,
)
from prismaquant.cb_anchored_cost import (
    CBUnitDeclaration,
    CodebookAnchoredFormatPlugin,
    LATTICE_BASIS,
    anchors_from_streamed_payload,
    build_cb_allocator_cost_payload,
    build_cb_units,
    fit_all_cb_segments,
    fitted_cb_hull_report,
    merge_streamed_cb_anchor_aura_shards,
    observations_from_streamed_payload,
    run_streamed_cb_anchor_aura,
)
from prismaquant.cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json_sha256,
)
from prismaquant.production_cache_stripes import plan_stripes
from prismaquant.rtx4090_qwen38_policy import (
    RTX4090_CONTEXT_FIRST_TARGET_BYTES,
    RTX4090_QWEN38_FORMAT_MENU,
    RTX4090_QWEN38_SERVING_PROFILE,
    validate_rtx4090_format_menu,
)


PLAN_SCHEMA = "prismaquant.rtx4090_fp8_burn.plan.v1"
MERGED_SCHEMA = "prismaquant.rtx4090_fp8_burn.merged_aura.v1"
ALLOCATOR_COST_SCHEMA = "prismaquant.rtx4090_fp8_burn.allocator_cost.v1"
STRIPE_COUNT = 2
STREAMING_CACHE_MAX_SLOTS = 2
ARTIFACT_OVERHEAD_RESERVE_BYTES = 268_435_456
BF16_FORMAT = "BF16"
NATIVE_FP8_FORMAT = "FP8_E4M3"
CB_FORMATS = tuple(RTX4090_QWEN38_FORMAT_MENU[:-2])
MEASURED_CB_FORMATS = (CB_FORMATS[0], CB_FORMATS[3], CB_FORMATS[-1])
ANCHOR_CB_FORMAT = CB_FORMATS[3]
MEASURED_FORMATS = (*MEASURED_CB_FORMATS, NATIVE_FP8_FORMAT)
FULL_FORMATS = (*CB_FORMATS, NATIVE_FP8_FORMAT, BF16_FORMAT)
RENDER_FORMATS = (*MEASURED_FORMATS, BF16_FORMAT)
RENDER_LEVERS: Mapping[str, object] = {
    "gptq": True,
    "static_act_order": True,
    "joint_scale_opt": True,
    "weighted_vq": True,
}
CB_PRODUCER_SETTINGS: Mapping[str, object] = {
    "scale_coding": "v1",
    "codebook_source": "lattice",
    "codebook_source_scope": "none",
    "scale_sweep": True,
    "scale_sweep_scope": "fp8",
    "ldlq": False,
    "ldlq_scope": "none",
    "minchain": False,
    "encode_tier": "balanced",
    "activation_scope": "none",
    "encode_compile": True,
    "atom_compile": True,
}

_LAYER_RE = re.compile(r"(?:^|[.])layers[.](\d+)(?:[.]|$)")


class RTX4090FP8BurnError(ValueError):
    """An input or receipt differs from the immutable campaign contract."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _binding(path: str | Path, *, where: str) -> dict[str, object]:
    """Bind an input without copying path-dependent or unrelated contents."""
    input_path = Path(path)
    if not input_path.is_file():
        raise RTX4090FP8BurnError(f"{where} is not a file: {input_path}")
    result: dict[str, object] = {
        "sha256": _sha256_file(input_path),
        "bytes": input_path.stat().st_size,
    }
    if input_path.suffix.lower() == ".json":
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RTX4090FP8BurnError(
                f"{where} is not valid JSON: {input_path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RTX4090FP8BurnError(f"{where} must be a JSON object")
        schema = payload.get("schema")
        if schema is not None:
            result["schema"] = str(schema)
        for key in (
            "content_sha256", "resolved_commit", "source_sha256",
            "tree_sha256", "closure_sha256",
        ):
            value = payload.get(key)
            if value is not None:
                result[key] = str(value)
    return result


def _probe_payload(path: str | Path) -> tuple[dict[str, dict], dict[str, Any]]:
    try:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)  # trusted local pipeline artifact
    except Exception as exc:
        raise RTX4090FP8BurnError(f"probe is unreadable: {path}") from exc
    stats = payload.get("stats") if isinstance(payload, Mapping) else None
    meta = payload.get("meta") if isinstance(payload, Mapping) else None
    if not isinstance(stats, Mapping) or not isinstance(meta, Mapping):
        raise RTX4090FP8BurnError("probe lacks stats/meta mappings")
    return (
        {str(name): dict(row) for name, row in stats.items()
         if isinstance(row, Mapping)},
        dict(meta),
    )


def _calibration_contract(
    meta: Mapping[str, object], *, nsamples: int, seqlen: int, seed: int,
) -> dict[str, object]:
    for name, expected in (("nsamples", nsamples), ("seqlen", seqlen)):
        if int(meta.get(name, -1)) != int(expected):
            raise RTX4090FP8BurnError(
                f"probe {name}={meta.get(name)!r}, expected {expected}"
            )
    digest = meta.get("calib_hash")
    if not isinstance(digest, str) or not digest:
        raise RTX4090FP8BurnError("probe has no exact calib_hash")
    return {
        "calib_hash": digest,
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
        "seed": int(seed),
    }


def _validate_stats(stats: Mapping[str, Mapping[str, object]]) -> None:
    if not stats:
        raise RTX4090FP8BurnError("body probe census is empty")
    for qname, row in stats.items():
        n_params = int(row.get("n_params", 0) or 0)
        in_features = int(row.get("in_features", 0) or 0)
        out_features = int(row.get("out_features", 0) or 0)
        if min(n_params, in_features, out_features) <= 0:
            raise RTX4090FP8BurnError(f"{qname}: incomplete positive shape")
        if n_params != in_features * out_features:
            raise RTX4090FP8BurnError(
                f"{qname}: n_params differs from in_features*out_features"
            )


def _qname_maps(qnames: Sequence[str]) -> dict[str, object]:
    ordered = tuple(sorted(str(name) for name in qnames))
    return {
        "formats_by_qname": {name: list(RENDER_FORMATS) for name in ordered},
        "purposes_by_qname": {
            name: {
                MEASURED_CB_FORMATS[0]: ["panel"],
                ANCHOR_CB_FORMAT: ["anchor", "panel"],
                MEASURED_CB_FORMATS[-1]: ["panel"],
                NATIVE_FP8_FORMAT: ["anchor"],
            }
            for name in ordered
        },
        "unmeasured_formats_by_qname": {
            name: [BF16_FORMAT] for name in ordered
        },
        "legal_cb_formats_by_qname": {
            name: list(CB_FORMATS) for name in ordered
        },
    }


def _stripe_metrics(
    qnames: Sequence[str], stats: Mapping[str, Mapping[str, object]],
) -> dict[str, int]:
    return {
        "qnames": len(qnames),
        "parameters": sum(int(stats[name]["n_params"]) for name in qnames),
        "estimated_work": sum(
            int(stats[name]["n_params"])
            * max(int(stats[name]["in_features"]), 1)
            for name in qnames
        ),
        "render_cells": len(qnames) * len(MEASURED_FORMATS),
    }


def _campaign_stripes(
    stats: Mapping[str, Mapping[str, object]], *, profile,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Resolve two LPT bins, preferring an exactly tied contiguous split.

    For the periodic 64-layer Qwen body, contiguous halves are another
    optimal LPT tie resolution.  Prefer them only after proving that their
    qname, parameter, work, and render-cell totals are exactly equal and that
    those totals match the ordinary deterministic LPT solution.
    """
    lpt = plan_stripes(stats, profile=profile, n_stripes=STRIPE_COUNT)
    lpt_metrics = tuple(_stripe_metrics(stripe.qnames, stats) for stripe in lpt)
    selected_qnames = tuple(tuple(stripe.qnames) for stripe in lpt)
    selected_groups = tuple(tuple(stripe.groups) for stripe in lpt)
    strategy = "whole_layer_lpt"
    layer_ranges: tuple[list[int] | None, ...] = (None, None)

    by_layer: dict[int, list[str]] = {}
    non_layer: list[str] = []
    for qname in sorted(stats):
        match = _LAYER_RE.search(qname)
        if match is None:
            non_layer.append(qname)
        else:
            by_layer.setdefault(int(match.group(1)), []).append(qname)
    layers = tuple(sorted(by_layer))
    if not non_layer and len(layers) == 64 and layers == tuple(range(64)):
        halves = (layers[:32], layers[32:])
        contiguous = tuple(
            tuple(sorted(name for layer in half for name in by_layer[layer]))
            for half in halves
        )
        contiguous_metrics = tuple(_stripe_metrics(names, stats) for names in contiguous)
        metric_names = ("qnames", "parameters", "estimated_work", "render_cells")
        exactly_equal = all(
            contiguous_metrics[0][name] == contiguous_metrics[1][name]
            for name in metric_names
        )
        same_as_lpt = sorted(
            tuple(row[name] for name in metric_names) for row in contiguous_metrics
        ) == sorted(tuple(row[name] for name in metric_names) for row in lpt_metrics)
        if exactly_equal and same_as_lpt:
            selected_qnames = contiguous
            selected_groups = tuple(
                tuple(f"layer:{layer}" for layer in half) for half in halves
            )
            strategy = "whole_layer_lpt_contiguous_equal_tie"
            layer_ranges = ([0, 31], [32, 63])

    selected_metrics = tuple(_stripe_metrics(names, stats) for names in selected_qnames)
    records = tuple({
        "index": index,
        "qnames": list(selected_qnames[index]),
        "groups": list(selected_groups[index]),
        "estimated_work": selected_metrics[index]["estimated_work"],
        "parameters": selected_metrics[index]["parameters"],
        "render_cells": selected_metrics[index]["render_cells"],
        "layer_range": layer_ranges[index],
        "qname_file": f"stripe-{index:02d}.qnames.txt",
        "qname_file_sha256": hashlib.sha256("".join(
            f"{name}\n" for name in selected_qnames[index]
        ).encode("utf-8")).hexdigest(),
    } for index in range(STRIPE_COUNT))
    proof = {
        "strategy": strategy,
        "metric_names": ["qnames", "parameters", "estimated_work", "render_cells"],
        "selected": [dict(row) for row in selected_metrics],
        "ordinary_lpt": [dict(row) for row in lpt_metrics],
        "selected_metrics_exactly_equal": selected_metrics[0] == selected_metrics[1],
        "selected_matches_lpt_loads": sorted(selected_metrics, key=lambda row: tuple(row.values()))
        == sorted(lpt_metrics, key=lambda row: tuple(row.values())),
    }
    return records, proof


def _fixed_census_records(
    fixed_bf16: Mapping[str, Mapping[str, object] | str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for qname in sorted(fixed_bf16):
        raw = fixed_bf16[qname]
        if isinstance(raw, Mapping):
            reason = str(raw.get("reason", "fixed_bf16"))
            record = {
                "qname": qname,
                "format": BF16_FORMAT,
                "reason": reason,
                "source_dtype": str(raw.get("source_dtype", "bf16")),
                "n_params": int(raw.get("n_params", 0) or 0),
            }
        else:
            record = {
                "qname": qname,
                "format": BF16_FORMAT,
                "reason": str(raw),
                "source_dtype": "bf16",
                "n_params": 0,
            }
        if record["source_dtype"] != "bf16":
            raise RTX4090FP8BurnError(
                f"fixed unit {qname} is not a BF16 source"
            )
        records.append(record)
    return records


def build_campaign_plan(
    stats: Mapping[str, Mapping[str, object]],
    *,
    profile,
    fixed_bf16: Mapping[str, Mapping[str, object] | str],
    calibration: Mapping[str, object],
    bindings: Mapping[str, Mapping[str, object]],
    source_dtype_census_sha256: str,
) -> dict[str, object]:
    """Build the path-independent global plan used by both GPU hosts."""
    validate_rtx4090_format_menu(FULL_FORMATS)
    body_stats = {str(name): dict(row) for name, row in stats.items()}
    _validate_stats(body_stats)
    fixed_records = _fixed_census_records(fixed_bf16)
    overlap = sorted(set(body_stats) & {str(row["qname"]) for row in fixed_records})
    if overlap:
        raise RTX4090FP8BurnError(
            f"body and fixed-BF16 censuses overlap: {overlap[:8]}"
        )
    stripe_records, balance_proof = _campaign_stripes(
        body_stats, profile=profile
    )
    maps = _qname_maps(tuple(sorted(body_stats)))
    plan: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "policy": {
            "serving_profile": RTX4090_QWEN38_SERVING_PROFILE,
            "target_bytes": RTX4090_CONTEXT_FIRST_TARGET_BYTES,
            "artifact_overhead_reserve_bytes": ARTIFACT_OVERHEAD_RESERVE_BYTES,
            "formats": list(FULL_FORMATS),
            "rendered_formats": list(RENDER_FORMATS),
            "measured_formats": list(MEASURED_FORMATS),
            "measured_cb_anchors": list(MEASURED_CB_FORMATS),
            "primary_cb_anchor": ANCHOR_CB_FORMAT,
            "unmeasured_terminal": BF16_FORMAT,
            "codebook_formats": list(CB_FORMATS),
            "lattice_only": True,
        },
        "producer": {
            "cb_serialization": dict(CB_PRODUCER_SETTINGS),
            "render_levers": dict(RENDER_LEVERS),
            "transient_renders": True,
            "purpose": "anchor",
            "streamed_model_cache": {
                "max_cache_slots": STREAMING_CACHE_MAX_SLOTS,
                "effective_prefetch_lookahead": 0,
            },
        },
        "calibration": dict(calibration),
        "bindings": {str(k): dict(v) for k, v in sorted(bindings.items())},
        "source_dtype_census_sha256": str(source_dtype_census_sha256),
        "body": {
            "qnames": list(sorted(body_stats)),
            "qname_count": len(body_stats),
            "parameters": sum(
                int(row["n_params"]) for row in body_stats.values()
            ),
            "shapes": {
                name: [int(body_stats[name]["out_features"]),
                       int(body_stats[name]["in_features"])]
                for name in sorted(body_stats)
            },
        },
        "fixed_bf16_census": fixed_records,
        "fixed_bf16_census_sha256": canonical_json_sha256(
            fixed_records, where="RTX4090 fixed BF16 census"
        ),
        "stripes": stripe_records,
        "stripe_balance_proof": balance_proof,
        "maps": maps,
    }
    plan["plan_sha256"] = canonical_json_sha256(
        plan, where="RTX4090 FP8 burn plan"
    )
    validate_campaign_plan(plan)
    return plan


def validate_campaign_plan(plan: Mapping[str, object]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise RTX4090FP8BurnError("campaign plan schema mismatch")
    expected_digest = plan.get("plan_sha256")
    without_digest = dict(plan)
    without_digest.pop("plan_sha256", None)
    observed_digest = canonical_json_sha256(
        without_digest, where="RTX4090 FP8 burn plan"
    )
    if expected_digest != observed_digest:
        raise RTX4090FP8BurnError("campaign plan digest mismatch")
    policy = plan.get("policy")
    body = plan.get("body")
    maps = plan.get("maps")
    stripes = plan.get("stripes")
    if not all(isinstance(item, Mapping) for item in (policy, body, maps)):
        raise RTX4090FP8BurnError("campaign plan lacks policy/body/maps")
    if not isinstance(stripes, Sequence) or isinstance(stripes, (str, bytes)):
        raise RTX4090FP8BurnError("campaign plan lacks stripe records")
    assert isinstance(policy, Mapping) and isinstance(body, Mapping)
    assert isinstance(maps, Mapping)
    if tuple(policy.get("formats", ())) != FULL_FORMATS:
        raise RTX4090FP8BurnError("campaign plan format menu is not exact")
    if tuple(policy.get("rendered_formats", ())) != RENDER_FORMATS:
        raise RTX4090FP8BurnError("campaign sparse render menu is not exact")
    if tuple(policy.get("measured_formats", ())) != MEASURED_FORMATS:
        raise RTX4090FP8BurnError("campaign measured menu is not exact")
    qnames = tuple(str(name) for name in body.get("qnames", ()))
    if qnames != tuple(sorted(qnames)) or len(qnames) != len(set(qnames)):
        raise RTX4090FP8BurnError("body qnames are not unique and sorted")
    expected_maps = _qname_maps(qnames)
    if dict(maps) != expected_maps:
        raise RTX4090FP8BurnError("campaign qname maps are not the exact menu")
    if len(stripes) != STRIPE_COUNT:
        raise RTX4090FP8BurnError("campaign must contain exactly two stripes")
    flattened: list[str] = []
    owners: dict[str, int] = {}
    for expected_index, raw in enumerate(stripes):
        if not isinstance(raw, Mapping) or int(raw.get("index", -1)) != expected_index:
            raise RTX4090FP8BurnError("stripe indices are not canonical")
        members = [str(name) for name in raw.get("qnames", ())]
        text = "".join(f"{name}\n" for name in members).encode("utf-8")
        if raw.get("qname_file_sha256") != hashlib.sha256(text).hexdigest():
            raise RTX4090FP8BurnError("stripe qname file binding mismatch")
        flattened.extend(members)
        for name in members:
            layer_groups = tuple(
                group for group in raw.get("groups", ())
                if str(group).startswith("layer:")
            )
            if layer_groups:
                owners[name] = expected_index
    if len(flattened) != len(set(flattened)) or set(flattened) != set(qnames):
        raise RTX4090FP8BurnError("stripes are not an exact disjoint body cover")
    # Every LPT decoder group is written once; this catches hand-edited plans
    # even when the flattened qname cover still looks complete.
    all_groups = [str(group) for raw in stripes if isinstance(raw, Mapping)
                  for group in raw.get("groups", ())]
    if len(all_groups) != len(set(all_groups)):
        raise RTX4090FP8BurnError("a whole-layer group spans stripes")
    proof = plan.get("stripe_balance_proof")
    if not isinstance(proof, Mapping):
        raise RTX4090FP8BurnError("campaign lacks stripe balance proof")
    selected = proof.get("selected")
    if not isinstance(selected, Sequence) or len(selected) != STRIPE_COUNT:
        raise RTX4090FP8BurnError("stripe balance proof is incomplete")
    for index, raw in enumerate(stripes):
        assert isinstance(raw, Mapping)
        metrics = selected[index]
        if not isinstance(metrics, Mapping):
            raise RTX4090FP8BurnError("stripe balance metrics are malformed")
        if (
            int(metrics.get("qnames", -1)) != len(raw.get("qnames", ()))
            or int(metrics.get("parameters", -1)) != int(raw.get("parameters", -2))
            or int(metrics.get("estimated_work", -1)) != int(raw.get("estimated_work", -2))
            or int(metrics.get("render_cells", -1)) != int(raw.get("render_cells", -2))
        ):
            raise RTX4090FP8BurnError("stripe balance proof differs from stripe")
    if proof.get("strategy") == "whole_layer_lpt_contiguous_equal_tie":
        if proof.get("selected_metrics_exactly_equal") is not True:
            raise RTX4090FP8BurnError("contiguous tie is not exactly balanced")
        if proof.get("selected_matches_lpt_loads") is not True:
            raise RTX4090FP8BurnError("contiguous tie differs from LPT loads")
        if [raw.get("layer_range") for raw in stripes if isinstance(raw, Mapping)] != [
            [0, 31], [32, 63]
        ]:
            raise RTX4090FP8BurnError("contiguous tie ranges are not 0-31/32-63")


def write_campaign_plan(plan: Mapping[str, object], output_dir: str | Path) -> Path:
    validate_campaign_plan(plan)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for raw in plan["stripes"]:  # type: ignore[index]
        assert isinstance(raw, Mapping)
        path = root / str(raw["qname_file"])
        data = "".join(f"{name}\n" for name in raw["qnames"]).encode("utf-8")
        if hashlib.sha256(data).hexdigest() != raw["qname_file_sha256"]:
            raise RTX4090FP8BurnError("refusing inconsistent stripe qname file")
        atomic_write_bytes(path, data)
    plan_path = root / "campaign-plan.json"
    atomic_write_bytes(plan_path, json.dumps(
        plan, indent=2, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8") + b"\n")
    return plan_path


def load_campaign_plan(path: str | Path) -> dict[str, object]:
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RTX4090FP8BurnError(f"campaign plan is unreadable: {path}") from exc
    if not isinstance(plan, Mapping):
        raise RTX4090FP8BurnError("campaign plan is not an object")
    result = dict(plan)
    validate_campaign_plan(result)
    return result


def _classify_source_census(
    source_census: Mapping[str, str], *, profile,
) -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
    body: list[str] = []
    fixed: dict[str, dict[str, object]] = {}
    for qname in sorted(source_census):
        source_dtype = str(source_census[qname]).lower()
        if source_dtype != "bf16":
            raise RTX4090FP8BurnError(
                f"source Linear {qname} has dtype class {source_dtype!r}; "
                "this campaign requires one exact BF16 source class"
            )
        if profile.is_pinned_name(qname):
            reason = "profile_pinned"
        elif qname.startswith("mtp.") or ".mtp." in qname:
            reason = "mtp_fixed"
        elif qname.startswith("visual.") or ".visual." in qname:
            reason = "visual_fixed"
        else:
            body.append(qname)
            continue
        fixed[qname] = {"reason": reason, "source_dtype": source_dtype}
    return tuple(body), fixed


def prepare(args: argparse.Namespace) -> Path:
    from prismaquant.allocator_candidates import _scan_source_dtype_manifest
    from prismaquant.model_profiles import detect_profile

    profile = detect_profile(args.model)
    if profile.name not in {"qwen3_5_dense", "qwen3_5"}:
        raise RTX4090FP8BurnError(
            f"model profile {profile.name!r} is not the dense Qwen source"
        )
    probe_stats, probe_meta = _probe_payload(args.probe)
    source_census = _scan_source_dtype_manifest(args.model, profile)
    body, fixed = _classify_source_census(source_census, profile=profile)
    missing_probe = sorted(set(body) - set(probe_stats))
    extra_probe = sorted(set(probe_stats) - set(body) - set(fixed))
    if missing_probe or extra_probe:
        raise RTX4090FP8BurnError(
            "probe/source census mismatch: "
            f"missing body={missing_probe[:8]}, extra={extra_probe[:8]}"
        )
    body_stats = {name: probe_stats[name] for name in body}
    for name in fixed:
        if name in probe_stats:
            fixed[name]["n_params"] = int(probe_stats[name].get("n_params", 0) or 0)
        else:
            fixed[name]["n_params"] = 0
    calibration = _calibration_contract(
        probe_meta, nsamples=args.n_calib_samples,
        seqlen=args.calib_seqlen, seed=args.calib_seed,
    )
    bindings = {
        "probe": _binding(args.probe, where="probe"),
        "col_weights": _binding(args.col_weights, where="column weights"),
        "source_model_identity": _binding(
            args.source_identity, where="source model identity"
        ),
        "producer_snapshot": _binding(
            args.producer_snapshot, where="producer source snapshot"
        ),
        "common_execution_attestation": _binding(
            args.execution_attestation, where="common execution attestation"
        ),
        "dataset": _binding(args.dataset, where="calibration dataset"),
    }
    census_digest = canonical_json_sha256(
        dict(sorted((str(k), str(v)) for k, v in source_census.items())),
        where="source dtype census",
    )
    plan = build_campaign_plan(
        body_stats, profile=profile, fixed_bf16=fixed,
        calibration=calibration, bindings=bindings,
        source_dtype_census_sha256=census_digest,
    )
    return write_campaign_plan(plan, args.output_dir)


def _verify_binding(plan: Mapping[str, object], name: str, path: str | Path) -> None:
    bindings = plan.get("bindings")
    expected = bindings.get(name) if isinstance(bindings, Mapping) else None
    if not isinstance(expected, Mapping):
        raise RTX4090FP8BurnError(f"plan has no {name} binding")
    observed = _binding(path, where=name)
    if observed != dict(expected):
        raise RTX4090FP8BurnError(f"{name} differs from the prepared plan")


def _cb_context():
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    return CBSerializationContext(
        scale_coding="v1", codebook_source="lattice",
        codebook_source_scope="none", scale_sweep=True,
        scale_sweep_scope="fp8", ldlq=False, ldlq_scope="none",
        minchain=False, encode_tier="balanced", activation_contract=None,
        activation_execution=None,
    )


def _arm_identity(plan: Mapping[str, object]) -> dict[str, object]:
    bindings = plan["bindings"]
    assert isinstance(bindings, Mapping)
    producer_snapshot = bindings["producer_snapshot"]
    execution = bindings["common_execution_attestation"]
    assert isinstance(producer_snapshot, Mapping)
    assert isinstance(execution, Mapping)
    return {
        "campaign_schema": PLAN_SCHEMA,
        "global_plan_sha256": plan["plan_sha256"],
        "producer_settings": dict(CB_PRODUCER_SETTINGS),
        "render_levers": dict(RENDER_LEVERS),
        "producer_snapshot_sha256": producer_snapshot["sha256"],
        "common_execution_attestation_sha256": execution["sha256"],
        "compile_settings": {
            "PRISMAQUANT_CB_ENCODE_COMPILE": "1",
            "PRISMAQUANT_CB_ATOM_COMPILE": "1",
        },
        "sparse_anchor_measurement": True,
    }


def _require_compile_settings() -> dict[str, str]:
    names = (
        "PRISMAQUANT_CB_ENCODE_COMPILE",
        "PRISMAQUANT_CB_ATOM_COMPILE",
    )
    resolved = {name: str(os.environ.get(name, "")).strip() for name in names}
    disabled = [name for name, value in resolved.items() if value != "1"]
    if disabled:
        raise RTX4090FP8BurnError(
            "strict CB compilation is not enabled: "
            f"set {', '.join(disabled)}=1 before measuring"
        )
    return resolved


def measure(args: argparse.Namespace) -> Path:
    """Run exactly one stripe; this is the only GPU-bearing entry point."""
    plan = load_campaign_plan(args.plan)
    compile_settings = _require_compile_settings()
    stripe_index = int(args.stripe)
    if stripe_index not in range(STRIPE_COUNT):
        raise RTX4090FP8BurnError("stripe must be 0 or 1")
    for name, path in (
        ("probe", args.probe), ("col_weights", args.col_weights),
        ("source_model_identity", args.source_identity),
        ("producer_snapshot", args.producer_snapshot),
        ("common_execution_attestation", args.execution_attestation),
        ("dataset", args.dataset),
    ):
        _verify_binding(plan, name, path)

    from prismaquant.build_production_cache import _load_col_weights
    from prismaquant.cost_streaming import (
        build_streamed_causal_lm, build_streamed_model_identity,
    )
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.model_profiles import detect_profile
    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.sensitivity_probe import load_calibration
    import torch
    from transformers import AutoTokenizer

    device = require_cuda_hot_path("rtx4090_fp8_burn", "cuda")
    profile = detect_profile(args.model)
    probe_stats, probe_meta = _probe_payload(args.probe)
    calibration_contract = plan["calibration"]
    assert isinstance(calibration_contract, Mapping)
    if _calibration_contract(
        probe_meta, nsamples=int(calibration_contract["nsamples"]),
        seqlen=int(calibration_contract["seqlen"]),
        seed=int(calibration_contract["seed"]),
    ) != dict(calibration_contract):
        raise RTX4090FP8BurnError("probe calibration identity changed")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        local_files_only=Path(args.model).exists(),
    )
    calibration = load_calibration(
        tokenizer, args.dataset, int(calibration_contract["nsamples"]),
        int(calibration_contract["seqlen"]),
        calib_seed=int(calibration_contract["seed"]),
    ).to(device)
    observed_hash = calibration_data_hash(calibration)
    if observed_hash != calibration_contract["calib_hash"]:
        raise RTX4090FP8BurnError(
            "tokenized calibration differs from the prepared probe"
        )
    col_weights = _load_col_weights(args.col_weights, CB_FORMATS)
    if not isinstance(col_weights, Mapping):
        raise RTX4090FP8BurnError("column weights did not load")
    body_qnames = tuple(plan["body"]["qnames"])  # type: ignore[index]
    missing_col = sorted(set(body_qnames) - set(col_weights))
    if missing_col:
        raise RTX4090FP8BurnError(
            f"column weights miss body units: {missing_col[:8]}"
        )
    stripe = plan["stripes"][stripe_index]  # type: ignore[index]
    stripe_qnames = tuple(stripe["qnames"])
    stripe_stats = {name: probe_stats[name] for name in stripe_qnames}
    activation_index = ActivationIndex(Path(args.activation_cache_dir), stripe_stats)
    missing_act = [name for name in stripe_qnames if name not in activation_index]
    if missing_act:
        raise RTX4090FP8BurnError(
            f"activation cache misses stripe units: {missing_act[:8]}"
        )
    maps = plan["maps"]
    formats = {name: tuple(maps["formats_by_qname"][name])
               for name in stripe_qnames}
    purposes = {name: maps["purposes_by_qname"][name]
                for name in stripe_qnames}
    legal = {name: tuple(maps["legal_cb_formats_by_qname"][name])
             for name in stripe_qnames}
    checkpoint_root = Path(args.checkpoint_dir)
    runner = build_streamed_causal_lm(
        args.model, device=device, dtype=torch.bfloat16,
        offload_folder=str(checkpoint_root / "streamed-model-offload"),
        profile=profile, max_cache_slots=STREAMING_CACHE_MAX_SLOTS,
    )
    try:
        model_identity = build_streamed_model_identity(
            runner, args.model,
            identity_cache_path=checkpoint_root / "streamed_model_identity.json",
        )
        source_binding = plan["bindings"]["source_model_identity"]  # type: ignore[index]
        if model_identity.get("content_sha256") != source_binding.get("content_sha256"):
            raise RTX4090FP8BurnError(
                "live streamed source identity differs from prepared source"
            )
        arm_identity = _arm_identity(plan)
        if arm_identity["compile_settings"] != compile_settings:
            raise RTX4090FP8BurnError(
                "resolved compile switches differ from the campaign stamp"
            )
        payload = run_streamed_cb_anchor_aura(
            runner, calibration, formats_by_qname=formats,
            legal_formats_by_qname=legal, purposes_by_qname=purposes,
            activation_index=activation_index, render_levers=RENDER_LEVERS,
            col_weights=col_weights, cb_serialization_context=_cb_context(),
            calibration_hash=observed_hash, arm_identity=arm_identity,
            model_identity=model_identity,
            checkpoint_dir=checkpoint_root / "aura", resume=bool(args.resume),
            n_probes=int(args.n_probes), profile=profile,
            checkpoint_identity_extra={
                "campaign_schema": PLAN_SCHEMA,
                "global_plan_sha256": plan["plan_sha256"],
                "stripe_index": stripe_index,
                "stripe_qname_file_sha256": stripe["qname_file_sha256"],
                "fixed_bf16_census_sha256": plan["fixed_bf16_census_sha256"],
                "compile_settings": compile_settings,
                "streaming_source_cache": {
                    "max_cache_slots": STREAMING_CACHE_MAX_SLOTS,
                    "effective_prefetch_lookahead": 0,
                },
            },
        )
    finally:
        runner.shutdown()
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise RTX4090FP8BurnError(f"stripe output exists: {output}")
    if output.exists() and args.resume:
        with output.open("rb") as handle:
            prior = pickle.load(handle)
        if canonical_json_sha256(
            prior.get("costs", {}), where="prior stripe costs"
        ) != canonical_json_sha256(payload.get("costs", {}), where="stripe costs"):
            raise RTX4090FP8BurnError("completed stripe output differs on resume")
        return output
    atomic_write_bytes(output, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return output


def _load_pickle_mapping(path: str | Path, *, where: str) -> dict[str, object]:
    try:
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
    except Exception as exc:
        raise RTX4090FP8BurnError(f"{where} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise RTX4090FP8BurnError(f"{where} is not a mapping")
    return dict(value)


def merge(args: argparse.Namespace) -> Path:
    plan = load_campaign_plan(args.plan)
    _verify_binding(plan, "col_weights", args.col_weights)
    from prismaquant.build_production_cache import _load_col_weights

    col_weights = _load_col_weights(args.col_weights, CB_FORMATS)
    if not isinstance(col_weights, Mapping):
        raise RTX4090FP8BurnError("column weights did not load")
    payloads = [_load_pickle_mapping(path, where="AURA stripe")
                for path in args.shards]
    if len(payloads) != STRIPE_COUNT:
        raise RTX4090FP8BurnError("merge requires exactly two raw shards")
    body_qnames = tuple(plan["body"]["qnames"])  # type: ignore[index]
    maps = plan["maps"]
    merged = merge_streamed_cb_anchor_aura_shards(
        payloads, col_weights=col_weights, expected_qnames=body_qnames,
        expected_formats_by_qname=maps["formats_by_qname"],
        expected_purposes_by_qname=maps["purposes_by_qname"],
        expected_unmeasured_formats_by_qname=(
            maps["unmeasured_formats_by_qname"]
        ),
        expected_legal_cb_formats_by_qname=(
            maps["legal_cb_formats_by_qname"]
        ),
    )
    costs = merged.get("costs")
    if not isinstance(costs, Mapping) or set(costs) != set(body_qnames):
        raise RTX4090FP8BurnError("merged cost qnames are not the full plan")
    for qname in body_qnames:
        rows = costs[qname]
        if not isinstance(rows, Mapping) or set(rows) != set(MEASURED_FORMATS):
            raise RTX4090FP8BurnError(f"{qname}: merged measured menu differs")
        native = rows[NATIVE_FP8_FORMAT]
        if (
            not isinstance(native, Mapping)
            or native.get("cost_source") != "aura"
            or native.get("production_anchor_measured") is not True
        ):
            raise RTX4090FP8BurnError(
                f"{qname}: native FP8 is not a fresh direct measured row"
            )
    provenance = merged.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise RTX4090FP8BurnError("merged payload provenance is not mutable")
    provenance["rtx4090_fp8_burn"] = {
        "schema": MERGED_SCHEMA,
        "global_plan_sha256": plan["plan_sha256"],
        "fixed_bf16_census_sha256": plan["fixed_bf16_census_sha256"],
        "producer_snapshot_sha256": plan["bindings"]["producer_snapshot"]["sha256"],  # type: ignore[index]
        "common_execution_attestation_sha256": plan["bindings"]["common_execution_attestation"]["sha256"],  # type: ignore[index]
        "direct_measured_formats": list(MEASURED_FORMATS),
        "unmeasured_terminal": BF16_FORMAT,
    }
    output = Path(args.output)
    if output.exists():
        raise RTX4090FP8BurnError(f"merged output exists: {output}")
    atomic_write_bytes(output, pickle.dumps(merged, protocol=pickle.HIGHEST_PROTOCOL))
    return output


def _allocator_cost(
    merged: Mapping[str, object], plan: Mapping[str, object],
) -> dict[str, object]:
    result = copy.deepcopy(dict(merged))
    costs = result.get("costs")
    if not isinstance(costs, dict):
        raise RTX4090FP8BurnError("merged payload lacks mutable costs")
    for qname in plan["body"]["qnames"]:  # type: ignore[index]
        rows = costs.get(qname)
        if not isinstance(rows, dict) or set(rows) != set(MEASURED_FORMATS):
            raise RTX4090FP8BurnError(f"{qname}: allocator input menu differs")
        native_before = copy.deepcopy(rows[NATIVE_FP8_FORMAT])
        rows[BF16_FORMAT] = {
            "predicted_dloss": 0.0,
            "output_mse_measured": False,
            "cost_source": "aura_passthrough_zero",
            "unmeasured_terminal_identity": True,
        }
        if rows[NATIVE_FP8_FORMAT] != native_before:
            raise RTX4090FP8BurnError("native FP8 row changed during finalization")
    provenance = result.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise RTX4090FP8BurnError("allocator cost provenance is not mutable")
    provenance["rtx4090_fp8_burn_allocator_cost"] = {
        "schema": ALLOCATOR_COST_SCHEMA,
        "global_plan_sha256": plan["plan_sha256"],
        "target_bytes": RTX4090_CONTEXT_FIRST_TARGET_BYTES,
        "bf16_terminal_is_identity_by_construction": True,
    }
    return result


def allocate(args: argparse.Namespace) -> Path:
    plan = load_campaign_plan(args.plan)
    _verify_binding(plan, "probe", args.probe)
    _verify_binding(plan, "col_weights", args.col_weights)
    merged = _load_pickle_mapping(args.merged, where="merged AURA payload")
    burn = merged.get("provenance", {}).get("rtx4090_fp8_burn")
    if not isinstance(burn, Mapping) or burn.get("global_plan_sha256") != plan["plan_sha256"]:
        raise RTX4090FP8BurnError("merged AURA payload is not bound to plan")
    cost_payload = _allocator_cost(merged, plan)
    cost_path = Path(args.cost_output)
    if cost_path.exists():
        raise RTX4090FP8BurnError(f"allocator cost output exists: {cost_path}")
    atomic_write_bytes(
        cost_path, pickle.dumps(cost_payload, protocol=pickle.HIGHEST_PROTOCOL)
    )
    output_dir = Path(args.output_dir)
    command = [
        sys.executable, "-m", "prismaquant.allocator",
        "--probe", str(args.probe), "--costs", str(cost_path),
        "--model-override", str(args.model),
        "--target-profile", RTX4090_QWEN38_SERVING_PROFILE,
        "--target-disk-gb", "18.000000000",
        "--artifact-overhead-reserve-bytes",
        str(ARTIFACT_OVERHEAD_RESERVE_BYTES),
        "--cb-scale-coding", "v1", "--cb-codebook-source", "lattice",
        "--cb-codebook-source-scope", "none", "--cb-scale-sweep", "1",
        "--cb-scale-sweep-scope", "fp8", "--cb-ldlq", "0",
        "--cb-ldlq-scope", "none", "--cb-encode-tier", "balanced",
        "--cb-col-weights", str(args.col_weights),
        "--formats", ",".join(FULL_FORMATS),
        "--lm-head-format", BF16_FORMAT, "--mtp-format", BF16_FORMAT,
        "--visual-format", BF16_FORMAT, "--threads", str(args.threads),
        "--layer-config", str(output_dir / "layer_config.json"),
        "--pareto-csv", str(output_dir / "pareto.csv"),
        "--pareto-output-dir", str(output_dir / "pareto-points"),
        "--applicability-report", str(output_dir / "format_applicability.json"),
        "--bit-attribution-json", str(output_dir / "bit_attribution.json"),
        "--bit-attribution-csv", str(output_dir / "bit_attribution.csv"),
    ]
    run_allocator_once(
        command=command, output_dir=output_dir, resume=bool(args.resume),
        invocation_provenance={
            "campaign_schema": PLAN_SCHEMA,
            "global_plan_sha256": plan["plan_sha256"],
            "target_bytes": RTX4090_CONTEXT_FIRST_TARGET_BYTES,
            "cost_payload_sha256": _sha256_file(cost_path),
            "direct_native_fp8_measurement": True,
            "bf16_unmeasured_terminal": True,
        },
    )
    return output_dir / "layer_config.json"


def _common_prepare(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--col-weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-identity", required=True)
    parser.add_argument("--producer-snapshot", required=True)
    parser.add_argument("--execution-attestation", required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    _common_prepare(prepare_parser)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--n-calib-samples", type=int, default=32)
    prepare_parser.add_argument("--calib-seqlen", type=int, default=1024)
    prepare_parser.add_argument("--calib-seed", type=int, default=42)
    prepare_parser.set_defaults(handler=prepare)

    measure_parser = sub.add_parser("measure")
    _common_prepare(measure_parser)
    measure_parser.add_argument("--plan", required=True)
    measure_parser.add_argument("--stripe", required=True, type=int)
    measure_parser.add_argument("--activation-cache-dir", required=True)
    measure_parser.add_argument("--checkpoint-dir", required=True)
    measure_parser.add_argument("--output", required=True)
    measure_parser.add_argument("--n-probes", type=int, default=32)
    measure_parser.add_argument("--resume", action="store_true")
    measure_parser.set_defaults(handler=measure)

    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--plan", required=True)
    merge_parser.add_argument("--col-weights", required=True)
    merge_parser.add_argument("--shards", nargs="+", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.set_defaults(handler=merge)

    allocate_parser = sub.add_parser("allocate")
    allocate_parser.add_argument("--plan", required=True)
    allocate_parser.add_argument("--model", required=True)
    allocate_parser.add_argument("--probe", required=True)
    allocate_parser.add_argument("--col-weights", required=True)
    allocate_parser.add_argument("--merged", required=True)
    allocate_parser.add_argument("--cost-output", required=True)
    allocate_parser.add_argument("--output-dir", required=True)
    allocate_parser.add_argument("--threads", type=int, default=16)
    allocate_parser.add_argument("--resume", action="store_true")
    allocate_parser.set_defaults(handler=allocate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = args.handler(args)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
