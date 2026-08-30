#!/usr/bin/env python3
"""Balanced fixed-vs-learned FP8-CB screen on the finalized GLM corpus.

This driver binds an explicit ``trellis.bf16_corpus.v2`` manifest to the
provenance-locked 2026-08-29 ``fp8_ladder.py`` implementation.  It measures
only K32/K40/K48, always renders both fixed and per-tensor learned books at
``encode_tier='balanced'``, and reports dense and routed populations apart.

It is research scaffolding: no GPU result from this driver is a serving,
runtime, or promotion claim.  GPU campaigns still require the repository's
in-process profiler plus Netdata/power evidence on both hosts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

EXPECTED_FP8_LADDER_SHA256 = (
    "f9c5167905b98fe98a3389a9471cb9bea06e6ced9a1288329ce1b0fb6a92d2a3"
)
EXPECTED_HULL_SWEEP_SHA256 = (
    "4420108cae7b024ae7effa75111a187efc0018220082ba724bf995c62b902a98"
)
DEFAULT_LOCKED_LADDER = Path("/home/rob/dq-runs/trellis-hull-20260828/fp8_ladder.py")
RUNGS = (32, 40, 48)
ENCODE_TIER = "balanced"
SCHEMA = "trellis.glm_fp8_learned_balanced.v1"


class CampaignError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _identity_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _locked_sources(ladder_path: Path) -> dict[str, object]:
    ladder = ladder_path.resolve(strict=True)
    hull = ladder.parent / "hull_sweep.py"
    got_ladder = _sha256(ladder)
    got_hull = _sha256(hull)
    if got_ladder != EXPECTED_FP8_LADDER_SHA256:
        raise CampaignError(
            f"locked fp8_ladder.py hash differs: {got_ladder} != "
            f"{EXPECTED_FP8_LADDER_SHA256}"
        )
    if got_hull != EXPECTED_HULL_SWEEP_SHA256:
        raise CampaignError(
            f"locked hull_sweep.py hash differs: {got_hull} != "
            f"{EXPECTED_HULL_SWEEP_SHA256}"
        )
    source = ladder.read_text()
    for symbol in (
        "def cb_arm_fp8(",
        "def cb_arm_fp8_learned(",
        "def assert_legal_e4m3_book(",
        "def fp8_cb_payload(",
    ):
        if symbol not in source:
            raise CampaignError(f"locked ladder is missing {symbol}")
    return {
        "fp8_ladder_path": str(ladder),
        "fp8_ladder_sha256": got_ladder,
        "hull_sweep_path": str(hull.resolve()),
        "hull_sweep_sha256": got_hull,
    }


def _load_ladder(ladder_path: Path):
    parent = str(ladder_path.resolve().parent)
    sys.path.insert(0, parent)
    try:
        module = importlib.import_module("fp8_ladder")
    finally:
        sys.path.pop(0)
    if Path(inspect.getsourcefile(module) or "").resolve() != ladder_path.resolve():
        raise CampaignError("fp8_ladder import escaped the locked source path")
    for function_name in ("cb_arm_fp8", "cb_arm_fp8_learned"):
        signature = inspect.signature(getattr(module, function_name))
        if "encode_tier" not in signature.parameters:
            raise CampaignError(f"{function_name} has no explicit encode_tier")
    return module


def population_summaries(per_tensor: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Summarize each GLM population independently; never emit a pooled row."""

    output: dict[str, object] = {}
    for population in ("dense", "routed"):
        names = sorted(
            name for name, cell in per_tensor.items()
            if cell.get("population") == population
        )
        if not names:
            continue
        rows = []
        for rung in RUNGS:
            deltas = []
            fixed_db = []
            learned_db = []
            fixed_bpw = []
            learned_bpw = []
            for name in names:
                arms = per_tensor[name]["arms"]
                fixed = arms[f"fp8_cb@{rung}"]
                learned = arms[f"fp8_cb_learned@{rung}"]
                fixed_db.append(float(fixed["weighted_snr_db"]))
                learned_db.append(float(learned["weighted_snr_db"]))
                fixed_bpw.append(float(fixed["footprint"]["exact_bpw"]))
                learned_bpw.append(float(learned["footprint"]["exact_bpw"]))
                deltas.append(learned_db[-1] - fixed_db[-1])
            rows.append({
                "rung": rung,
                "tensors": len(names),
                "fixed_db_median": statistics.median(fixed_db),
                "learned_db_median": statistics.median(learned_db),
                "learned_minus_fixed_db_median": statistics.median(deltas),
                "learned_minus_fixed_db_min": min(deltas),
                "learned_minus_fixed_db_max": max(deltas),
                "learned_better": sum(delta > 0 for delta in deltas),
                "fixed_bpw_median": statistics.median(fixed_bpw),
                "learned_bpw_median": statistics.median(learned_bpw),
            })
        output[population] = {"tensors": len(names), "rows": rows}
    return output


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.write-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locked-ladder", type=Path, default=DEFAULT_LOCKED_LADDER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from prismaquant.trellis_bf16_corpus import load_finalized_bf16_corpus
    corpus = load_finalized_bf16_corpus(args.manifest)
    locked = _locked_sources(args.locked_ladder)
    settings = {
        "schema": SCHEMA,
        "corpus_manifest": str(corpus.manifest_path),
        "corpus_file_sha256": corpus.manifest["file_sha256"],
        "importance_value_sha256": corpus.manifest["importance_identity"]["value_sha256"],
        "population_counts": {
            name: len(entries) for name, entries in corpus.populations.items()
        },
        "rungs": list(RUNGS),
        "encode_tier": ENCODE_TIER,
        "locked_sources": locked,
        "aggregation_contract": "dense/routed population-separated; no pooled median",
    }
    settings["identity_sha256"] = _identity_sha256(settings)
    if args.dry_run:
        print(json.dumps({**settings, "status": "validated_no_gpu_no_write"},
                         indent=2, sort_keys=True))
        return 0
    if args.out.exists():
        raise CampaignError("final output already exists (immutable no-clobber)")

    import torch
    if not torch.cuda.is_available():
        raise CampaignError("CUDA required for learned FP8-CB encoding")
    ladder = _load_ladder(args.locked_ladder)
    partial = args.out.with_name(args.out.name + ".partial")
    if partial.exists():
        report = json.loads(partial.read_text())
        if report.get("settings") != settings:
            raise CampaignError("partial checkpoint identity differs")
    else:
        report = {
            "schema": SCHEMA,
            "settings": settings,
            "started_at_unix_s": time.time(),
            "per_tensor": {},
            "partial": True,
        }
    per_tensor = report["per_tensor"]
    device = torch.device("cuda")
    for index, entry in enumerate(corpus.entries, 1):
        if entry.name in per_tensor:
            continue
        raw, importance = corpus.load_tensor(entry)
        weight = raw.to(device=device, dtype=torch.float32)
        metric = importance.to(device=device, dtype=torch.float32).reshape(1, -1)
        energy = ladder.C.weighted_sse(
            weight, torch.zeros_like(weight), metric
        )
        if not math.isfinite(energy) or energy <= 0:
            raise CampaignError(f"{entry.name}: invalid weighted energy")
        cell = {
            "population": entry.population,
            "shape": list(weight.shape),
            "source_weight_sha256": entry.source_weight_sha256,
            "importance_sha256": entry.importance_sha256,
            "importance_source": {
                "qname": entry.importance_source_qname,
                "expert": entry.importance_source_expert,
                "denominator_name": entry.importance_denominator_name,
                "denominator": entry.importance_denominator,
            },
            "weighted_energy": energy,
            "arms": {},
        }
        for rung in RUNGS:
            fixed, seconds = ladder.timed(
                lambda k=rung: ladder.cb_arm_fp8(
                    weight, k, metric, encode_tier=ENCODE_TIER
                ),
                sync=True,
            )
            (learned, tables), learned_seconds = ladder.timed(
                lambda k=rung: ladder.cb_arm_fp8_learned(
                    weight, k, metric, encode_tier=ENCODE_TIER
                ),
                sync=True,
            )
            book = ladder.assert_legal_e4m3_book(tables, rung)
            for arm, reconstruction, elapsed, footprint, extra in (
                (
                    f"fp8_cb@{rung}", fixed, seconds,
                    ladder.fp8_cb_payload(list(weight.shape), rung), {},
                ),
                (
                    f"fp8_cb_learned@{rung}", learned, learned_seconds,
                    ladder.fp8_cb_payload(
                        list(weight.shape), rung, learned=True,
                        book_shapes=[tuple(table.shape) for table in tables],
                    ),
                    {"learned_book": book},
                ),
            ):
                error = ladder.C.weighted_sse(weight, reconstruction, metric)
                cell["arms"][arm] = {
                    "encode_tier": ENCODE_TIER,
                    "encode_seconds_observation_not_perf_claim": elapsed,
                    "weighted_sse": error,
                    "weighted_nsse": error / energy,
                    "weighted_snr_db": -10.0 * math.log10(max(error / energy, 1e-300)),
                    "reconstruction_sha256": ladder.tensor_sha256(
                        reconstruction.contiguous()
                    ),
                    "footprint": footprint,
                    **extra,
                }
        per_tensor[entry.name] = cell
        report["tensors_done"] = len(per_tensor)
        _atomic_json(partial, report)
        print(f"[{index}/{len(corpus.entries)}] {entry.population} {entry.name}",
              flush=True)
        del raw, importance, weight, metric
        torch.cuda.empty_cache()

    report.update({
        "partial": False,
        "completed_at_unix_s": time.time(),
        "population_summaries": population_summaries(per_tensor),
        "status": "measurement_complete_no_serving_verdict",
        "performance_gate": (
            "encode timings are observations only; attach in-process profiler "
            "and both-host Netdata/power evidence before any performance claim"
        ),
    })
    _atomic_json(partial, report)
    os.rename(partial, args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
