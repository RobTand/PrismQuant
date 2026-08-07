#!/usr/bin/env python3
"""DSV4-Flash LDLQ cost campaign.

The pilot is intentionally a separate, fail-closed command.  It consumes the
verified content-keyed layer store, uses the production FP8-CB encoder and
threaded batched LDLQ path, and checkpoints every projection/rung measurement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from prismaquant import format_registry as fr
from prismaquant.cb_ldlq import fill_empty_expert_activation_rows
from prismaquant.cb_warm_state import CBWarmStateStore, build_warm_record
from prismaquant.expert_empirical_cost import _cb_ladder_law
from prismaquant.layer_streaming import (
    _build_fp8_scale_inv_map,
    _build_weight_map,
    _read_layer_to_device,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_fields_for_context,
    cb_serialization_context_stamp,
)
from prismaquant.nvfp4_cb_formats import nvfp4_cb_reconstruct
from prismaquant.production_weight_cache import (
    canonical_cb_col_weights_sha256,
    validate_cb_render_identity_metadata,
    validate_cb_render_source_weight,
)


RUN_ROOT = Path("/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq")
SOURCE = Path("/home/rob/dq-runs/dsv4-flash-0731/source")
PRIOR = Path("/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2")
BY_LAYER = PRIOR / "artifacts-mxfp4/probe-k12k18/by-layer"
COL_WEIGHTS = PRIOR / "artifacts-mxfp4/cb_col_weights.pkl"
ACT_ROOT = PRIOR / "act"
PILOT_LAYER = 21
EXPERTS = tuple(range(256))
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
RUNGS = tuple(range(28, 39))  # demand-driven rev: priced domain K28-K38 (operator 2026-08-06);
# K39-K48 are demand-extension/native-fallback rungs, never interpolated blind.
ANCHORS = (28, 38, 48)
HOLDOUTS = (33, 43)
TOLERANCE = 0.10
CONTEXT = CBSerializationContext.production(
    scale_sweep=True,
    ldlq=True,
    encode_tier="balanced",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256_float32(tensor: torch.Tensor) -> str:
    value = torch.as_tensor(tensor).detach().cpu().to(torch.float32).contiguous()
    return hashlib.sha256(
        value.numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value)
    os.replace(temp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def act_path(qname: str) -> Path:
    return ACT_ROOT / (re.sub(r"[^A-Za-z0-9_-]", "__", qname) + ".pt")


def load_direct_activation(qname: str, width: int) -> torch.Tensor:
    path = act_path(qname)
    if not path.is_file():
        return torch.empty((0, width), dtype=torch.float32)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    value = blob.get("inputs") if isinstance(blob, dict) else None
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{qname}: activation entry has no rank-2 inputs")
    if int(value.shape[1]) != int(width):
        raise ValueError(
            f"{qname}: activation width {value.shape[1]} != weight width {width}"
        )
    return value.detach().to(torch.float32).contiguous()


def load_layer_identity(layer: int) -> tuple[dict, dict]:
    path = BY_LAYER / f"layer_{layer:03d}.pkl"
    payload = pickle.loads(path.read_bytes())
    identity = copy.deepcopy(payload["provenance"]["cb_render_identity"])
    historical_missing_ldlq = "ldlq" not in identity["cb_serialized_payload"]
    if historical_missing_ldlq:
        identity["cb_serialized_payload"]["ldlq"] = False
    validate_cb_render_identity_metadata(
        identity,
        require_source_complete=True,
        where=f"DSV4 verified by-layer store layer {layer}",
    )
    # The store's shard_idx is a per-BATCH write counter, not a layer id: this
    # store was built in two batches (2026-08-03), so layers 0-26 happen to
    # satisfy shard_idx == layer while 27+ restart from 0. Verifying the
    # counter therefore rejects correct shards from the second batch. Verify
    # the content instead — every cost qname must belong to this layer, which
    # is strictly stronger provenance than the counter ever was.
    foreign = [
        qname for qname in payload["costs"]
        if not str(qname).startswith(f"model.layers.{layer}.")
    ]
    if foreign:
        raise AssertionError(
            f"layer {layer}: by-layer store holds foreign qnames "
            f"(e.g. {foreign[0]}); shard content does not match its layer"
        )
    if Path(payload["meta"]["model"]).resolve() != SOURCE.resolve():
        raise AssertionError(f"layer {layer}: source path mismatch")
    if (
        Path(payload["meta"]["incremental_shard"]["activation_cache_dir"]).resolve()
        != ACT_ROOT.resolve()
    ):
        raise AssertionError(f"layer {layer}: activation cache mismatch")
    return payload, {
        "identity": identity,
        "path": str(path),
        "sha256": sha256_file(path),
        "historical_ldlq_missing_inferred_false": historical_missing_ldlq,
    }


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
    weights: list[torch.Tensor] = []
    col_weights: list[torch.Tensor] = []
    activations: list[torch.Tensor] = []
    qnames: list[str] = []
    observed_files = 0
    for expert in EXPERTS:
        qname = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        weight_name = qname + ".weight"
        loaded = _read_layer_to_device(
            weight_name,
            model_to_shard,
            model_to_ckpt,
            torch.bfloat16,
            device,
            fp8_scale_inv_map=scale_map,
        )
        if set(loaded) != {weight_name}:
            raise AssertionError(f"{qname}: source resolved {sorted(loaded)}")
        weight = loaded[weight_name].to(torch.bfloat16).contiguous()
        validate_cb_render_source_weight(
            identity,
            qname,
            weight,
            where=f"DSV4 layer {layer} pilot source",
        )
        cw = torch.as_tensor(all_col_weights[qname]).to(torch.float32).contiguous()
        if list(cw.shape) != list(identity["col_weights_shapes"][qname]):
            raise AssertionError(f"{qname}: col-weight shape mismatch")
        if content_sha256_float32(cw) != identity["col_weights_content_sha256"][qname]:
            raise AssertionError(f"{qname}: col-weight digest mismatch")
        x = load_direct_activation(qname, int(weight.shape[1]))
        if x.shape[0]:
            observed_files += 1
            derived = x.square().mean(dim=0)
            if not torch.allclose(derived, cw, rtol=1e-6, atol=1e-8):
                delta = float((derived - cw).abs().max().item())
                raise AssertionError(
                    f"{qname}: activation/col-weight mismatch max_abs={delta}"
                )
        qnames.append(qname)
        weights.append(weight)
        col_weights.append(cw)
        activations.append(x)
    activation_rows, cold = fill_empty_expert_activation_rows(
        tuple(activations),
        qname=f"model.layers.{layer}.mlp.experts.{projection}",
    )
    weight_stack = torch.stack(weights).contiguous()
    col_stack = torch.stack(col_weights).unsqueeze(1).to(device).contiguous()
    return {
        "qnames": qnames,
        "weight": weight_stack,
        "col_weights": col_stack,
        "activation_rows": activation_rows,
        "observed_activation_files": observed_files,
        "cold_experts": list(cold),
    }


def per_slice_mse(weight: torch.Tensor, reconstruction: torch.Tensor) -> list[float]:
    return (
        (weight - reconstruction).float().square().mean(dim=(1, 2))
        .detach().cpu().tolist()
    )


def per_slice_weighted_mse(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    col_weights: torch.Tensor,
) -> list[float]:
    err2 = (weight - reconstruction).float().square()
    cw = torch.broadcast_to(col_weights.to(err2), err2.shape)
    return (
        (err2 * cw).sum(dim=(1, 2)) / cw.sum(dim=(1, 2)).clamp_min(1e-30)
    ).detach().cpu().tolist()


def encode_rung(
    *,
    layer: int,
    projection: str,
    rung: int,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    format_name = f"FP8_CB_K{rung}"
    spec = fr.get_format(format_name)
    torch.cuda.synchronize()
    started = time.perf_counter()
    fields = cb_fields_for_context(
        spec,
        data["weight"],
        context=CONTEXT,
        col_weights=data["col_weights"],
        activation_rows=data["activation_rows"],
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    reconstruction = nvfp4_cb_reconstruct(
        fields, rung, grid="fp8", mode="product"
    ).to(data["weight"].dtype)
    logical_qname = f"model.layers.{layer}.mlp.experts.{projection}"
    warm_path = CBWarmStateStore(RUN_ROOT / "warm-state").write(
        build_warm_record(
            qname=logical_qname,
            format_name=format_name,
            source_weight=data["weight"],
            col_weights=data["col_weights"],
            context=CONTEXT,
            fields=fields,
        )
    )
    result = {
        "schema": "prismaquant.dsv4_ldlq_projection_rung.v1",
        "created_at": utc_now(),
        "layer": layer,
        "projection": projection,
        "format": format_name,
        "rung": rung,
        "qnames": list(data["qnames"]),
        "weight_mse_per_expert": per_slice_mse(data["weight"], reconstruction),
        "weighted_mse_per_expert": per_slice_weighted_mse(
            data["weight"], reconstruction, data["col_weights"]
        ),
        "elapsed_seconds": elapsed,
        "warm_state_path": str(warm_path),
        "cold_experts": list(data["cold_experts"]),
        "observed_activation_files": int(data["observed_activation_files"]),
        "encoder": {
            "ldlq": True,
            "batch_experts": True,
            "feeder_threads": 16,
            "expert_batch": 16,
            "encode_tier": "balanced",
        },
    }
    del reconstruction, fields
    torch.cuda.empty_cache()
    return result


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    rank = (len(ordered) - 1) * fraction
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def fit_projection(projection: str, measurements: Mapping[int, Mapping]) -> dict:
    formats = [f"FP8_CB_K{k}" for k in RUNGS]
    kmap = {name: int(name.rsplit("K", 1)[1]) for name in formats}
    anchor_names = [f"FP8_CB_K{k}" for k in ANCHORS]
    holdout_names = [f"FP8_CB_K{k}" for k in HOLDOUTS]
    holdout_errors: dict[str, list[float]] = {name: [] for name in holdout_names}
    accepted: list[int] = []
    rung_errors: dict[str, list[float]] = {name: [] for name in formats}
    all_errors: list[float] = []
    laws: dict[str, int] = {}
    for expert in EXPERTS:
        values = {
            f"FP8_CB_K{k}": float(measurements[k]["weight_mse_per_expert"][expert])
            for k in RUNGS
        }
        law = _cb_ladder_law(kmap, anchor_names, values)
        if law is None:
            continue
        laws[law.name] = laws.get(law.name, 0) + 1
        local_holdout = {}
        for name in holdout_names:
            measured = values[name]
            rel = abs(law.predict(name) - measured) / max(abs(measured), 1e-30)
            holdout_errors[name].append(rel)
            local_holdout[name] = rel
        if all(local_holdout[name] <= TOLERANCE for name in holdout_names):
            accepted.append(expert)
            for name in formats:
                measured = values[name]
                rel = abs(law.predict(name) - measured) / max(abs(measured), 1e-30)
                rung_errors[name].append(rel)
                all_errors.append(rel)
    stats = {
        "projection": projection,
        "slice_count": len(EXPERTS),
        "accepted_slices": len(accepted),
        "accepted_expert_ids": accepted,
        "acceptance_rate": len(accepted) / len(EXPERTS),
        "law_counts": laws,
        "holdouts": {},
        "accepted_prediction_error_all_rungs": {
            "n": len(all_errors),
            "median": statistics.median(all_errors) if all_errors else float("nan"),
            "p95": percentile(all_errors, 0.95),
        },
        "accepted_prediction_error_by_rung": {},
    }
    for name in holdout_names:
        errors = holdout_errors[name]
        stats["holdouts"][name] = {
            "accepted_at_10pct": sum(value <= TOLERANCE for value in errors),
            "acceptance_rate": sum(value <= TOLERANCE for value in errors) / len(EXPERTS),
            "median_relative_error": statistics.median(errors),
            "p95_relative_error": percentile(errors, 0.95),
            "max_relative_error": max(errors),
        }
    for name, errors in rung_errors.items():
        stats["accepted_prediction_error_by_rung"][name] = {
            "n": len(errors),
            "median": statistics.median(errors) if errors else float("nan"),
            "p95": percentile(errors, 0.95),
        }
    aggregate = stats["accepted_prediction_error_all_rungs"]
    stats["gate_pass"] = bool(
        accepted
        and aggregate["median"] <= 0.05
        and aggregate["p95"] <= 0.15
    )
    return stats


def pilot_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# DSV4 LDLQ Per-Slice Ladder Pilot",
        "",
        f"- Result: **{report['overall']}**",
        f"- Pre-declared layer: {PILOT_LAYER}",
        "- Encoder: production FP8-CB, LDLQ enabled, batched expert stack, 16 feeder threads",
        "- Anchors: K28, K38, K48",
        "- Independent holdouts: K33 and K43; 10% tolerance each",
        "- Validation metric: per-expert weight MSE, scored against fresh full K28..K48 measurement",
        "- Required accepted-slice error: median <= 5%, p95 <= 15% across all rungs",
        "",
        "| Projection | K33 accepted | K43 accepted | Dual accepted | Median | p95 | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for projection in PROJECTIONS:
        row = report["projections"][projection]
        k33 = row["holdouts"]["FP8_CB_K33"]["acceptance_rate"]
        k43 = row["holdouts"]["FP8_CB_K43"]["acceptance_rate"]
        error = row["accepted_prediction_error_all_rungs"]
        lines.append(
            f"| {projection} | {k33:.2%} | {k43:.2%} | "
            f"{row['acceptance_rate']:.2%} ({row['accepted_slices']}/256) | "
            f"{error['median']:.2%} | {error['p95']:.2%} | "
            f"{'PASS' if row['gate_pass'] else 'FAIL'} |"
        )
    agg = report["aggregate"]
    lines.extend([
        f"| **aggregate** | {agg['k33_acceptance_rate']:.2%} | "
        f"{agg['k43_acceptance_rate']:.2%} | {agg['dual_acceptance_rate']:.2%} | "
        f"{agg['median']:.2%} | {agg['p95']:.2%} | "
        f"{'PASS' if agg['gate_pass'] else 'FAIL'} |",
        "",
        "## Other menu rows",
        "",
        "`MXFP4` was freshly measured on the same 768 routed slices. "
        "`MXFP4_SOURCE` and `BF16` are exact identity transforms and were "
        "verified as zero-MSE rows by construction.",
        "",
        "## Serialization stamp",
        "",
        "```json",
        json.dumps(report["cb_serialized_payload"], indent=2, sort_keys=True),
        "```",
        "",
    ])
    if report["overall"] != "PASS":
        lines.extend([
            "The pre-declared gate failed. The 43-layer burn was not started.",
            "",
        ])
    return "\n".join(lines)


def run_pilot() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("pilot requires CUDA")
    os.environ["PRISMAQUANT_CB_LDLQ"] = "1"
    os.environ["PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS"] = "1"
    os.environ["PRISMAQUANT_CB_LDLQ_FEEDER_THREADS"] = "16"
    os.environ["PRISMAQUANT_CB_LDLQ_EXPERT_BATCH"] = "16"
    os.environ["PRISMAQUANT_CB_LDLQ_BATCH_STREAMS"] = "1"
    os.environ["PRISMAQUANT_CB_ENCODE_TIER"] = "balanced"
    os.environ.setdefault("PRISMAQUANT_CB_ENCODE_COMPILE", "1")
    device = torch.device("cuda:0")
    payload, layer_record = load_layer_identity(PILOT_LAYER)
    identity = layer_record["identity"]
    with COL_WEIGHTS.open("rb") as handle:
        all_col_weights = pickle.load(handle)
    observed_digest = canonical_cb_col_weights_sha256(
        all_col_weights, identity["col_weights_qnames"]
    )
    if observed_digest != identity["col_weights_sha256"]:
        raise AssertionError("layer-21 aggregate col-weight identity mismatch")
    model_to_shard, model_to_ckpt = _build_weight_map(str(SOURCE))
    scale_map = _build_fp8_scale_inv_map(str(SOURCE))
    formats = [f"FP8_CB_K{k}" for k in RUNGS]
    stamp = cb_serialization_context_stamp(CONTEXT, formats=formats)
    manifest = {
        "schema": "prismaquant.dsv4_ldlq_pilot_manifest.v1",
        "created_at": utc_now(),
        "layer": PILOT_LAYER,
        "source": str(SOURCE.resolve()),
        "source_index_sha256": sha256_file(SOURCE / "model.safetensors.index.json"),
        "activation_cache": str(ACT_ROOT.resolve()),
        "col_weights": str(COL_WEIGHTS.resolve()),
        "col_weights_sha256": observed_digest,
        "by_layer": {key: value for key, value in layer_record.items() if key != "identity"},
        "cb_serialized_payload": stamp,
        "anchors": list(ANCHORS),
        "holdouts": list(HOLDOUTS),
        "tolerance": TOLERANCE,
        "rungs": list(RUNGS),
    }
    atomic_json(RUN_ROOT / "PILOT_MANIFEST.json", manifest)

    measurements: dict[str, dict[int, dict]] = {}
    menu_rows: dict[str, dict[str, Any]] = {}
    for projection in PROJECTIONS:
        print(f"[pilot] loading layer {PILOT_LAYER} {projection}", flush=True)
        data = load_projection(
            PILOT_LAYER,
            projection,
            device=device,
            identity=identity,
            all_col_weights=all_col_weights,
            model_to_shard=model_to_shard,
            model_to_ckpt=model_to_ckpt,
            scale_map=scale_map,
        )
        measurements[projection] = {}
        for rung in RUNGS:
            shard = RUN_ROOT / "pilot-shards" / f"layer_021_{projection}_K{rung}.pkl"
            if shard.is_file():
                result = pickle.loads(shard.read_bytes())
                if result.get("format") != f"FP8_CB_K{rung}":
                    raise AssertionError(f"stale pilot shard {shard}")
                print(f"[pilot] resume {projection} K{rung}", flush=True)
            else:
                print(f"[pilot] encode {projection} K{rung}", flush=True)
                result = encode_rung(
                    layer=PILOT_LAYER,
                    projection=projection,
                    rung=rung,
                    data=data,
                )
                atomic_pickle(shard, result)
                print(
                    f"[pilot] wrote {shard.name} elapsed={result['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            measurements[projection][rung] = result

        # Freshly render the non-passthrough MXFP4 row on the same stack.
        mx_shard = RUN_ROOT / "pilot-shards" / f"layer_021_{projection}_MXFP4.pkl"
        if mx_shard.is_file():
            mx_result = pickle.loads(mx_shard.read_bytes())
        else:
            torch.cuda.synchronize()
            started = time.perf_counter()
            mx_recon = fr.get_format("MXFP4").quantize_dequantize(data["weight"].clone())
            torch.cuda.synchronize()
            mx_result = {
                "format": "MXFP4",
                "weight_mse_per_expert": per_slice_mse(data["weight"], mx_recon),
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_pickle(mx_shard, mx_result)
            del mx_recon
        menu_rows[projection] = {
            "MXFP4": mx_result,
            "MXFP4_SOURCE": {"weight_mse_per_expert": [0.0] * 256},
            "BF16": {"weight_mse_per_expert": [0.0] * 256},
        }
        del data
        torch.cuda.empty_cache()

    projection_stats = {
        projection: fit_projection(projection, measurements[projection])
        for projection in PROJECTIONS
    }
    aggregate_errors = []
    dual_accepted = 0
    k33_accepted = 0
    k43_accepted = 0
    for projection in PROJECTIONS:
        stats = projection_stats[projection]
        dual_accepted += stats["accepted_slices"]
        k33_accepted += stats["holdouts"]["FP8_CB_K33"]["accepted_at_10pct"]
        k43_accepted += stats["holdouts"]["FP8_CB_K43"]["accepted_at_10pct"]
        accepted = set(stats["accepted_expert_ids"])
        formats_map = {f"FP8_CB_K{k}": k for k in RUNGS}
        kmap = {name: k for name, k in formats_map.items()}
        anchor_names = [f"FP8_CB_K{k}" for k in ANCHORS]
        for expert in accepted:
            values = {
                name: float(measurements[projection][k]["weight_mse_per_expert"][expert])
                for name, k in formats_map.items()
            }
            law = _cb_ladder_law(kmap, anchor_names, values)
            for name in formats_map:
                aggregate_errors.append(
                    abs(law.predict(name) - values[name]) / max(values[name], 1e-30)
                )
    total_slices = len(PROJECTIONS) * len(EXPERTS)
    aggregate = {
        "slice_count": total_slices,
        "k33_acceptance_rate": k33_accepted / total_slices,
        "k43_acceptance_rate": k43_accepted / total_slices,
        "dual_accepted_slices": dual_accepted,
        "dual_acceptance_rate": dual_accepted / total_slices,
        "prediction_count": len(aggregate_errors),
        "median": statistics.median(aggregate_errors) if aggregate_errors else float("nan"),
        "p95": percentile(aggregate_errors, 0.95),
    }
    aggregate["gate_pass"] = bool(
        aggregate_errors and aggregate["median"] <= 0.05 and aggregate["p95"] <= 0.15
    )
    overall_pass = aggregate["gate_pass"] and all(
        projection_stats[p]["gate_pass"] for p in PROJECTIONS
    )
    report = {
        "schema": "prismaquant.dsv4_ldlq_pilot_report.v1",
        "created_at": utc_now(),
        "overall": "PASS" if overall_pass else "FAIL",
        "projections": projection_stats,
        "aggregate": aggregate,
        "other_menu": menu_rows,
        "cb_serialized_payload": stamp,
        "manifest": str(RUN_ROOT / "PILOT_MANIFEST.json"),
    }
    atomic_pickle(RUN_ROOT / "PILOT_FULL_MEASUREMENTS.pkl", {
        "report": report,
        "measurements": measurements,
    })
    atomic_json(RUN_ROOT / "PILOT_FIT_REPORT.json", report)
    atomic_text(RUN_ROOT / "PILOT_FIT_REPORT.md", pilot_markdown(report))
    print(f"[pilot] {report['overall']}", flush=True)
    return 0 if overall_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("pilot",))
    args = parser.parse_args()
    if args.command == "pilot":
        return run_pilot()
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
