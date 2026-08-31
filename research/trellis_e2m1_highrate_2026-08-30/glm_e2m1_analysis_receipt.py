#!/usr/bin/env python3
"""Verify and seal derived GLM E2M1 summaries against one final result.

The numeric campaign result is the authority.  This verifier independently
recomputes either the matched-rate coding-gain table or the near-four summary,
requires exact equality with the supplied derived JSON, and publishes a small
no-replace receipt binding both files and the tracked verifier closure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Mapping

import coding_gain_table
from atomic_publication import (
    PublicationError,
    atomic_checkpoint_json,
    canonical_json_bytes,
    exclusive_publication_claim,
    file_sha256,
    publish_file_no_replace,
)


SCHEMA = "trellis.glm_e2m1_analysis_receipt.v1"
NEAR_FOUR_SCHEMA = "trellis.glm_e2m1_near_four_summary.v1"
NEAR_FOUR_DEFINITION = (
    "scalar NVFP4 reconstructed exactly by summing each arm subset_split "
    "nvfp4_wsse over all scheduled column classes, divided by the cell "
    "weighted_energy; populations never pooled"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class AnalysisReceiptError(RuntimeError):
    pass


def _strict_json_object(path: Path) -> dict[str, object]:
    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str):
        raise ValueError(f"non-finite JSON constant {token!r}")

    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise AnalysisReceiptError(f"invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisReceiptError(f"{path} is not one JSON object")
    return value


def _identity_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _e2_checkpoint_sha256(document: Mapping[str, object]) -> str:
    body = {key: document[key] for key in ("receipt", "per_tensor")}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _validate_source(document: Mapping[str, object]) -> Mapping[str, object]:
    if set(document) != {"checkpoint_sha256", "receipt", "per_tensor"}:
        raise AnalysisReceiptError("source result field set differs")
    checkpoint = document.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or _SHA256.fullmatch(checkpoint) is None:
        raise AnalysisReceiptError("source checkpoint digest is invalid")
    if checkpoint != _e2_checkpoint_sha256(document):
        raise AnalysisReceiptError("source checkpoint self-digest differs")
    receipt = document.get("receipt")
    per_tensor = document.get("per_tensor")
    if not isinstance(receipt, dict) or not isinstance(per_tensor, dict):
        raise AnalysisReceiptError("source receipt/per_tensor is invalid")
    if (
        receipt.get("schema") != "trellis.e2m1_highrate.v3"
        or receipt.get("status") != "ok"
        or receipt.get("partial") is not False
        or receipt.get("corpus") != "glm"
        or receipt.get("glm_rate_plan") not in {"scaffold", "high"}
    ):
        raise AnalysisReceiptError("source is not a final GLM E2M1 v3 result")
    if receipt.get("tensors_done") != len(per_tensor) or not per_tensor:
        raise AnalysisReceiptError("source tensor completion differs")
    populations = {
        population: sum(
            isinstance(cell, dict) and cell.get("population") == population
            for cell in per_tensor.values()
        )
        for population in ("dense", "routed")
    }
    if any(count == 0 for count in populations.values()):
        raise AnalysisReceiptError("source must contain both GLM populations")
    if receipt.get("population_counts") != populations:
        raise AnalysisReceiptError("source population counts differ")
    publication = receipt.get("publication_identity_sha256")
    if not isinstance(publication, str) or _SHA256.fullmatch(publication) is None:
        raise AnalysisReceiptError("source publication identity is invalid")
    active = receipt.get("active_source_identity")
    environment = receipt.get("environment")
    if not isinstance(active, dict) or not isinstance(environment, dict):
        raise AnalysisReceiptError("source execution identity is absent")
    commit = active.get("repo_git_commit")
    if (
        not isinstance(commit, str)
        or _COMMIT.fullmatch(commit) is None
        or environment.get("repo_git_commit") != commit
    ):
        raise AnalysisReceiptError("source repository identity differs")
    return receipt


def _coding_gain_summary(
    source_path: Path, source: Mapping[str, object]
) -> dict[str, object]:
    per_tensor = source["per_tensor"]
    assert isinstance(per_tensor, dict)
    populations = sorted({
        str(cell["population"])
        for cell in per_tensor.values()
        if isinstance(cell, dict)
    })
    by_population = {}
    for population in populations:
        names = [
            name
            for name, cell in per_tensor.items()
            if isinstance(cell, dict) and cell.get("population") == population
        ]
        by_population[population] = {
            "n_tensors": len(names),
            "rows": coding_gain_table._summarize(per_tensor, names),
        }
    return {
        "schema": "trellis.coding_gain_table.v2",
        "definition": coding_gain_table.__doc__,
        "corpora": {
            "glm": {
                "label": coding_gain_table.LABEL["glm"],
                "source": str(source_path),
                "n_tensors": len(per_tensor),
                "aggregation_contract": "population-separated; no pooled median",
                "populations": by_population,
            }
        },
    }


def _near_four_summary(
    source_path: Path, source: Mapping[str, object]
) -> dict[str, object]:
    per_tensor = source["per_tensor"]
    assert isinstance(per_tensor, dict)
    output: dict[str, object] = {}
    for population in ("dense", "routed"):
        names = [
            name
            for name, cell in per_tensor.items()
            if isinstance(cell, dict) and cell.get("population") == population
        ]
        arm_names = {
            arm_name
            for name in names
            for arm_name in per_tensor[name]["arms"]
        }
        ordered_arms = sorted(
            arm_names,
            key=lambda item: (
                0 if item.startswith("tcq_two_tier@") else 1,
                float(item.split("@", 1)[1]),
            ),
        )
        rows = []
        for arm_name in ordered_arms:
            cells = [
                per_tensor[name]
                for name in names
                if arm_name in per_tensor[name]["arms"]
            ]
            trellis_db = [
                float(cell["arms"][arm_name]["weighted_snr_db"])
                for cell in cells
            ]
            scalar_db = []
            bpw = []
            for cell in cells:
                arm = cell["arms"][arm_name]
                scalar_wsse = sum(
                    float(split["nvfp4_wsse"])
                    for split in arm["subset_split"].values()
                )
                scalar_db.append(
                    -10.0
                    * math.log10(scalar_wsse / float(cell["weighted_energy"]))
                )
                bpw.append(float(arm["footprint"]["exact_bpw"]))
            paired = [
                trellis - scalar
                for trellis, scalar in zip(trellis_db, scalar_db)
            ]
            rows.append({
                "arm": arm_name,
                "complete": len(cells) == len(names),
                "exact_bpw_median": statistics.median(bpw),
                "paired_trellis_minus_scalar_db_max": max(paired),
                "paired_trellis_minus_scalar_db_median": statistics.median(paired),
                "paired_trellis_minus_scalar_db_min": min(paired),
                "population_tensors": len(names),
                "scalar_nvfp4_db_median": statistics.median(scalar_db),
                "tensors": len(cells),
                "trellis_db_median": statistics.median(trellis_db),
                "trellis_wins": sum(delta > 0.0 for delta in paired),
            })
        output[population] = {"rows": rows, "tensors": len(names)}
    return {
        "definition": NEAR_FOUR_DEFINITION,
        "populations": output,
        "schema": NEAR_FOUR_SCHEMA,
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
    }


def build_receipt(
    *, kind: str, source_path: Path, analysis_path: Path
) -> dict[str, object]:
    source_path = source_path.resolve(strict=True)
    analysis_path = analysis_path.resolve(strict=True)
    source = _strict_json_object(source_path)
    receipt = _validate_source(source)
    analysis = _strict_json_object(analysis_path)
    if kind == "coding_gain":
        expected = _coding_gain_summary(source_path, source)
        dependencies = [Path(coding_gain_table.__file__).resolve(strict=True)]
    elif kind == "near_four":
        expected = _near_four_summary(source_path, source)
        dependencies = []
    else:
        raise AnalysisReceiptError(f"unsupported analysis kind {kind!r}")
    if analysis != expected:
        raise AnalysisReceiptError(
            f"{kind} analysis differs from exact source recomputation"
        )
    verifier = Path(__file__).resolve(strict=True)
    commit = receipt["active_source_identity"]["repo_git_commit"]
    body: dict[str, object] = {
        "schema": SCHEMA,
        "status": "verified_exact_recomputation",
        "kind": kind,
        "source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "checkpoint_sha256": source["checkpoint_sha256"],
            "publication_identity_sha256": receipt["publication_identity_sha256"],
            "repo_git_commit": commit,
            "schema": receipt["schema"],
            "status": receipt["status"],
            "partial": receipt["partial"],
            "tensors_done": receipt["tensors_done"],
        },
        "analysis": {
            "path": str(analysis_path),
            "sha256": file_sha256(analysis_path),
            "schema": analysis["schema"],
        },
        "population_counts": receipt["population_counts"],
        "aggregation_contract": "dense/routed population-separated; no pooled median",
        "verifier": {
            "path": str(verifier),
            "sha256": file_sha256(verifier),
            "dependencies": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in dependencies
            ],
        },
    }
    return {**body, "receipt_sha256": _identity_sha256(body)}


def validate_receipt_self_digest(receipt: Mapping[str, object]) -> None:
    if set(receipt) != {
        "schema", "status", "kind", "source", "analysis",
        "population_counts", "aggregation_contract", "verifier",
        "receipt_sha256",
    }:
        raise AnalysisReceiptError("analysis receipt field set differs")
    digest = receipt.get("receipt_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise AnalysisReceiptError("analysis receipt digest is invalid")
    body = {key: value for key, value in receipt.items()
            if key != "receipt_sha256"}
    if digest != _identity_sha256(body):
        raise AnalysisReceiptError("analysis receipt self-digest differs")


def publish_receipt(output: Path, receipt: Mapping[str, object]) -> None:
    validate_receipt_self_digest(receipt)
    output = output.absolute()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or output.is_symlink():
        raise AnalysisReceiptError(f"receipt output already exists: {output}")
    if partial.exists() or partial.is_symlink():
        raise AnalysisReceiptError(f"receipt partial already exists: {partial}")
    try:
        with exclusive_publication_claim(output, identity=receipt):
            atomic_checkpoint_json(partial, receipt)
            publish_file_no_replace(partial, output)
    except PublicationError as exc:
        raise AnalysisReceiptError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("coding_gain", "near_four"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_receipt(
        kind=args.kind, source_path=args.source, analysis_path=args.analysis
    )
    publish_receipt(args.out, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
