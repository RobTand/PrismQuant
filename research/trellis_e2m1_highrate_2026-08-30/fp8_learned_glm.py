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
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]

from isolated_glm_corpus import load_active_glm_corpus
from atomic_publication import (
    PublicationError,
    atomic_checkpoint_json,
    exclusive_publication_claim,
    file_sha256,
    publish_file_no_replace,
)
from numeric_checkpoint_contract import (
    CheckpointContractError,
    FP8_PERFORMANCE_GATE,
    fp8_population_summaries,
    validate_fp8_checkpoint,
)

EXPECTED_FP8_LADDER_SHA256 = (
    "f9c5167905b98fe98a3389a9471cb9bea06e6ced9a1288329ce1b0fb6a92d2a3"
)
EXPECTED_HULL_SWEEP_SHA256 = (
    "4420108cae7b024ae7effa75111a187efc0018220082ba724bf995c62b902a98"
)
DEFAULT_LOCKED_LADDER = Path("/home/rob/dq-runs/trellis-hull-20260828/fp8_ladder.py")
RUNGS = (32, 40, 48)
ENCODE_TIER = "balanced"
SCHEMA = "trellis.glm_fp8_learned_balanced.v2"
CELL_KEYS = frozenset({
    "population", "shape", "source_weight_sha256", "importance_sha256",
    "importance_source", "weighted_energy", "arms",
})


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
    return fp8_population_summaries(per_tensor)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_checkpoint_json(path, value)


def _sealed_report(report: Mapping[str, object]) -> dict[str, object]:
    body = {key: value for key, value in report.items() if key != "checkpoint_sha256"}
    return {**body, "checkpoint_sha256": _identity_sha256(body)}


def _strict_json_object(path: Path) -> dict[str, object]:
    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CampaignError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(), object_pairs_hook=object_from_pairs)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CampaignError(f"invalid partial checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError("partial checkpoint must be one JSON object")
    return value


def _resume_report(path: Path, *, settings, corpus) -> dict[str, object]:
    sealed = _strict_json_object(path)
    body = {key: value for key, value in sealed.items()
            if key != "checkpoint_sha256"}
    if sealed.get("checkpoint_sha256") != _identity_sha256(body):
        raise CampaignError("partial checkpoint self-digest differs")
    try:
        validate_fp8_checkpoint(
            sealed, settings=settings, entries=corpus.entries
        )
    except CheckpointContractError as exc:
        raise CampaignError(f"partial checkpoint contract differs: {exc}") from exc
    return body


def _repo_commit() -> str | None:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit if len(commit) == 40 else None


def _active_source_identity() -> dict[str, object]:
    driver = Path(__file__).resolve(strict=True)
    isolated_loader = driver.with_name("isolated_glm_corpus.py")
    corpus_reader = REPO_ROOT / "prismaquant/trellis_bf16_corpus.py"
    return {
        "repo_root": str(REPO_ROOT),
        "repo_git_commit": _repo_commit(),
        "driver_path": str(driver),
        "driver_sha256": file_sha256(driver),
        "isolated_loader_path": str(isolated_loader),
        "isolated_loader_sha256": file_sha256(isolated_loader),
        "active_corpus_reader_path": str(corpus_reader.resolve(strict=True)),
        "active_corpus_reader_sha256": file_sha256(corpus_reader),
    }


def _frozen_codec_closure(ladder) -> dict[str, object]:
    hull = ladder.H
    imported = {}
    for name in ("H", "C", "W", "P", "S4", "TF"):
        module = getattr(ladder, name, None)
        module_path = Path(getattr(module, "__file__", ""))
        if module_path.is_file():
            resolved = module_path.resolve(strict=True)
            imported[name] = {
                "path": str(resolved),
                "sha256": file_sha256(resolved),
            }
    return {
        "snapshot_tree_sha256": hull.snapshot_tree_sha256(),
        "source_sha256": hull.source_hashes(),
        "imported_codec_modules": imported,
    }


def _verify_final_bindings(
    *, args, settings: Mapping[str, object], ladder
) -> None:
    if settings.get("active_source_identity") != _active_source_identity():
        raise CampaignError("active source identity drifted during run")
    if settings.get("locked_sources") != _locked_sources(args.locked_ladder):
        raise CampaignError("locked FP8/hull source identity drifted during run")
    if settings.get("frozen_codec_closure") != _frozen_codec_closure(ladder):
        raise CampaignError("frozen codec closure drifted during run")
    fresh = load_active_glm_corpus(REPO_ROOT, args.manifest)
    if (
        file_sha256(fresh.manifest_path)
        != settings.get("corpus_manifest_sha256")
        or fresh.manifest.get("file_sha256")
        != settings.get("corpus_file_sha256")
        or file_sha256(fresh.artifact_path)
        != settings.get("corpus_file_sha256")
        or fresh.manifest.get("importance_identity", {}).get("value_sha256")
        != settings.get("importance_value_sha256")
        or fresh.manifest.get("prismaquant_commit")
        != settings.get("corpus_prismaquant_commit")
    ):
        raise CampaignError("bound GLM corpus drifted during run")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locked-ladder", type=Path, default=DEFAULT_LOCKED_LADDER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    corpus = load_active_glm_corpus(REPO_ROOT, args.manifest)
    locked = _locked_sources(args.locked_ladder)
    ladder = _load_ladder(args.locked_ladder)
    settings = {
        "schema": SCHEMA,
        "corpus_manifest": str(corpus.manifest_path),
        "corpus_manifest_sha256": file_sha256(corpus.manifest_path),
        "corpus_file_sha256": corpus.manifest["file_sha256"],
        "importance_value_sha256": corpus.manifest["importance_identity"]["value_sha256"],
        "corpus_prismaquant_commit": corpus.manifest["prismaquant_commit"],
        "population_counts": {
            name: len(entries) for name, entries in corpus.populations.items()
        },
        "rungs": list(RUNGS),
        "encode_tier": ENCODE_TIER,
        "locked_sources": locked,
        "frozen_codec_closure": _frozen_codec_closure(ladder),
        "active_source_identity": _active_source_identity(),
        "aggregation_contract": "dense/routed population-separated; no pooled median",
    }
    settings["identity_sha256"] = _identity_sha256(settings)
    if args.dry_run:
        print(json.dumps({**settings, "status": "validated_no_gpu_no_write"},
                         indent=2, sort_keys=True))
        return 0
    try:
        with exclusive_publication_claim(args.out, identity=settings):
            return _run_claimed(args, corpus, settings, ladder)
    except PublicationError as exc:
        raise CampaignError(str(exc)) from exc


def _run_claimed(args, corpus, settings: Mapping[str, object], ladder) -> int:
    if args.out.exists() or args.out.is_symlink():
        raise CampaignError("final output already exists (immutable no-clobber)")

    import torch
    if not torch.cuda.is_available():
        raise CampaignError("CUDA required for learned FP8-CB encoding")
    partial = args.out.with_name(args.out.name + ".partial")
    if partial.is_symlink():
        raise CampaignError("partial checkpoint must not be a symlink")
    if partial.exists():
        report = _resume_report(partial, settings=settings, corpus=corpus)
    else:
        report = {
            "schema": SCHEMA,
            "settings": settings,
            "started_at_unix_s": time.time(),
            "per_tensor": {},
            "partial": True,
        }
    per_tensor = report["per_tensor"]
    generated_hashes: dict[str, dict[str, str]] = {}
    generated_books: dict[str, dict[str, object]] = {}
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
                reconstruction_hash = ladder.tensor_sha256(
                    reconstruction.contiguous()
                )
                cell["arms"][arm] = {
                    "encode_tier": ENCODE_TIER,
                    "encode_seconds_observation_not_perf_claim": elapsed,
                    "weighted_sse": error,
                    "weighted_nsse": error / energy,
                    "weighted_snr_db": -10.0 * math.log10(max(error / energy, 1e-300)),
                    "reconstruction_sha256": reconstruction_hash,
                    "footprint": footprint,
                    **extra,
                }
                generated_hashes.setdefault(entry.name, {})[arm] = (
                    reconstruction_hash
                )
                if "learned_book" in extra:
                    generated_books.setdefault(entry.name, {})[arm] = extra[
                        "learned_book"
                    ]
        per_tensor[entry.name] = cell
        report["tensors_done"] = len(per_tensor)
        sealed = _sealed_report(report)
        try:
            validate_fp8_checkpoint(
                sealed, settings=settings, entries=corpus.entries,
                generated_hashes=generated_hashes,
                generated_books=generated_books,
            )
        except CheckpointContractError as exc:
            raise CampaignError(
                f"refusing invalid generated checkpoint: {exc}"
            ) from exc
        _atomic_json(partial, sealed)
        print(f"[{index}/{len(corpus.entries)}] {entry.population} {entry.name}",
              flush=True)
        del raw, importance, weight, metric
        torch.cuda.empty_cache()

    report.update({
        "partial": False,
        "completed_at_unix_s": time.time(),
        "population_summaries": population_summaries(per_tensor),
        "status": "measurement_complete_no_serving_verdict",
        "performance_gate": FP8_PERFORMANCE_GATE,
    })
    sealed = _sealed_report(report)
    try:
        validate_fp8_checkpoint(
            sealed, settings=settings, entries=corpus.entries,
            require_partial=False,
            generated_hashes=generated_hashes,
            generated_books=generated_books,
        )
    except CheckpointContractError as exc:
        raise CampaignError(f"refusing invalid final result: {exc}") from exc
    _atomic_json(partial, sealed)
    _verify_final_bindings(args=args, settings=settings, ladder=ladder)
    try:
        publish_file_no_replace(partial, args.out)
    except PublicationError as exc:
        raise CampaignError(str(exc)) from exc
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
