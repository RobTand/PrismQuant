#!/usr/bin/env python3
"""Verify and seal derived GLM E2M1 summaries against one final result.

The numeric campaign result is the authority.  This verifier independently
recomputes either the matched-rate coding-gain table or the near-four summary,
requires exact equality with the supplied derived JSON, and publishes a small
no-replace receipt binding both files and the tracked verifier closure.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
from typing import Mapping

from atomic_publication import (
    PublicationError,
    atomic_checkpoint_json,
    canonical_json_bytes,
    exclusive_publication_claim,
    publish_file_no_replace,
)


SCHEMA = "trellis.glm_e2m1_analysis_receipt.v2"
NEAR_FOUR_SCHEMA = "trellis.glm_e2m1_near_four_summary.v1"
NEAR_FOUR_DEFINITION = (
    "scalar NVFP4 reconstructed exactly by summing each arm subset_split "
    "nvfp4_wsse over all scheduled column classes, divided by the cell "
    "weighted_energy; populations never pooled"
)
CODING_GAIN_DEFINITION = """The coding-gain table: trellis minus best-subset scalar at MATCHED rate,
restricted to the columns actually scheduled at that rate.

One derivation, both corpora, so the DSv4 and bf16 columns sit in one table
with nothing re-derived between them:

  dsv4 = DSv4 routed experts, MXFP4 source (21-27 distinct source values)
  bf16 = Qwen3-4B DENSE MLP, bf16 source (~5100 distinct source values)
         -- this corpus fixes the SOURCE-DTYPE confound and nothing else.
         It is dense Qwen3-4B: NOT MoE experts, NOT GLM.

The scalar partner is RTN onto the exhaustively-optimal 2^r-level subset of the
E2M1 grid, on the arm's OWN plane, with the same importance and the same
scorer.  Two fitting scopes are reported: `oracle` fits the subset on the rate-r
columns themselves (a TOUGHER baseline than the trellis gets, whose alphabet is
fit tensor-wide, so a positive oracle gain is conservative) and `shared` fits it
tensor-wide (the trellis alphabet's own scope).
"""
CODING_GAIN_LABEL = "GLM-5.3-Flash expert 0 + dense MLP, bf16 source"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MAX_BOUND_BYTES = 64 << 20
_EXPECTED_PLAN = {"coding_gain": "scaffold", "near_four": "high"}
_SUPERSESSION_REASON = (
    "v1 did not bind analysis kind to the exact GLM rate plan and parsed "
    "then hashed each input through separate path opens"
)


class AnalysisReceiptError(RuntimeError):
    pass


@dataclass(frozen=True)
class _BoundFile:
    path: Path
    raw: bytes
    sha256: str


def _bound_file(path: Path, *, maximum_bytes: int = _MAX_BOUND_BYTES) -> _BoundFile:
    """Read one regular non-symlink file through one stable descriptor.

    The returned bytes are the sole bytes parsed and hashed.  The final path
    identity is checked against the held descriptor after the read so an
    accidental rename/replacement during binding is a refusal.
    """

    if not hasattr(os, "O_NOFOLLOW"):
        raise AnalysisReceiptError("analysis binding requires O_NOFOLLOW")
    path = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AnalysisReceiptError(f"cannot open bound file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AnalysisReceiptError(f"bound path is not a regular file: {path}")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise AnalysisReceiptError(
                f"bound file size is outside 1..{maximum_bytes} bytes: {path}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(8 << 20, maximum_bytes + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum_bytes:
                raise AnalysisReceiptError(f"bound file exceeds size limit: {path}")
        after = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
            raise AnalysisReceiptError(f"bound file changed during read: {path}")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            raise AnalysisReceiptError(f"bound file read length differs: {path}")
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise AnalysisReceiptError(
                f"bound file path disappeared after read: {path}: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != after.st_dev
            or current.st_ino != after.st_ino
        ):
            raise AnalysisReceiptError(f"bound file path changed during read: {path}")
        return _BoundFile(
            path=path,
            raw=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    finally:
        os.close(descriptor)


def _bound_json_object(path: Path) -> tuple[dict[str, object], _BoundFile]:
    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str):
        raise ValueError(f"non-finite JSON constant {token!r}")

    bound = _bound_file(path)
    try:
        value = json.loads(
            bound.raw.decode("utf-8"),
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise AnalysisReceiptError(
            f"invalid JSON object {bound.path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AnalysisReceiptError(f"{bound.path} is not one JSON object")
    return value, bound


def _identity_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _e2_checkpoint_sha256(document: Mapping[str, object]) -> str:
    body = {key: document[key] for key in ("receipt", "per_tensor")}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _validate_source(
    document: Mapping[str, object], *, expected_plan: str
) -> Mapping[str, object]:
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
        or receipt.get("glm_rate_plan") != expected_plan
    ):
        raise AnalysisReceiptError(
            f"source is not a final GLM E2M1 v3 {expected_plan} result"
        )
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
            "rows": _coding_gain_rows(per_tensor, names),
        }
    return {
        "schema": "trellis.coding_gain_table.v2",
        "definition": CODING_GAIN_DEFINITION,
        "corpora": {
            "glm": {
                "label": CODING_GAIN_LABEL,
                "source": str(source_path),
                "n_tensors": len(per_tensor),
                "aggregation_contract": "population-separated; no pooled median",
                "populations": by_population,
            }
        },
    }


def _coding_gain_rows(per_tensor, names):
    rows = []
    for lane in ("tcq_two_tier", "tcq_v1"):
        keys = sorted(
            {
                key
                for name in names
                for key in per_tensor[name]["arms"]
                if key.startswith(f"{lane}@")
            },
            key=lambda key: float(key.split("@", 1)[1]),
        )
        for key in keys:
            rate = float(key.split("@", 1)[1])
            for column_class in (1, 2, 3, 4):
                cells = [
                    per_tensor[name]["arms"][key]["subset_split"][
                        str(column_class)
                    ]
                    for name in names
                    if key in per_tensor[name]["arms"]
                    and per_tensor[name]["arms"][key].get("subset_split")
                    and str(column_class)
                    in per_tensor[name]["arms"][key]["subset_split"]
                ]
                if len(cells) < len(names) / 2:
                    continue
                entry = {
                    "lane": lane,
                    "body_rate": rate,
                    "column_class": column_class,
                    "matched": abs(rate - column_class) < 1e-9,
                    "tensors": len(cells),
                    "columns_median": statistics.median(
                        cell["columns"] for cell in cells
                    ),
                    "trellis_db_median": statistics.median(
                        cell["trellis_db"] for cell in cells
                    ),
                    "nvfp4_db_median": statistics.median(
                        cell["nvfp4_db"] for cell in cells
                    ),
                }
                for scope in ("oracle", "shared"):
                    gains = [
                        cell[f"scalar_subgrid_{scope}"]["coding_gain_db"]
                        for cell in cells
                    ]
                    entry[f"scalar_{scope}_db_median"] = statistics.median(
                        cell[f"scalar_subgrid_{scope}"]["db"]
                        for cell in cells
                    )
                    entry[f"coding_gain_{scope}_db_median"] = statistics.median(
                        gains
                    )
                    entry[f"coding_gain_{scope}_db_min"] = min(gains)
                    entry[f"coding_gain_{scope}_db_max"] = max(gains)
                    entry[f"coding_gain_{scope}_positive"] = sum(
                        value > 0 for value in gains
                    )
                rows.append(entry)
    return rows


def _near_four_summary(
    source_path: Path,
    source: Mapping[str, object],
    *,
    source_sha256: str,
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
        "source_sha256": source_sha256,
    }


def _validate_superseded_v1(
    value: Mapping[str, object],
    *,
    kind: str,
    source_sha256: str,
    analysis_sha256: str,
) -> None:
    expected_fields = {
        "schema", "status", "kind", "source", "analysis",
        "population_counts", "aggregation_contract", "verifier",
        "receipt_sha256",
    }
    if set(value) != expected_fields:
        raise AnalysisReceiptError("superseded v1 receipt field set differs")
    if (
        value.get("schema") != "trellis.glm_e2m1_analysis_receipt.v1"
        or value.get("status") != "verified_exact_recomputation"
        or value.get("kind") != kind
    ):
        raise AnalysisReceiptError("superseded v1 receipt identity differs")
    digest = value.get("receipt_sha256")
    body = {key: item for key, item in value.items()
            if key != "receipt_sha256"}
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or digest != _identity_sha256(body)
    ):
        raise AnalysisReceiptError("superseded v1 receipt self-digest differs")
    source = value.get("source")
    analysis = value.get("analysis")
    if (
        not isinstance(source, dict)
        or source.get("sha256") != source_sha256
        or not isinstance(analysis, dict)
        or analysis.get("sha256") != analysis_sha256
    ):
        raise AnalysisReceiptError("superseded v1 receipt binds different inputs")


def build_receipt(
    *,
    kind: str,
    source_path: Path,
    analysis_path: Path,
    supersedes_path: Path | None = None,
) -> dict[str, object]:
    expected_plan = _EXPECTED_PLAN.get(kind)
    if expected_plan is None:
        raise AnalysisReceiptError(f"unsupported analysis kind {kind!r}")
    source, source_bound = _bound_json_object(source_path)
    receipt = _validate_source(source, expected_plan=expected_plan)
    analysis, analysis_bound = _bound_json_object(analysis_path)
    if kind == "coding_gain":
        expected = _coding_gain_summary(source_bound.path, source)
        dependency_paths = []
    elif kind == "near_four":
        expected = _near_four_summary(
            source_bound.path,
            source,
            source_sha256=source_bound.sha256,
        )
        dependency_paths = []
    if analysis != expected:
        raise AnalysisReceiptError(
            f"{kind} analysis differs from exact source recomputation"
        )
    verifier_bound = _bound_file(Path(__file__))
    dependency_bounds = [_bound_file(path) for path in dependency_paths]
    supersedes: dict[str, object] | None = None
    if supersedes_path is not None:
        superseded, superseded_bound = _bound_json_object(supersedes_path)
        _validate_superseded_v1(
            superseded,
            kind=kind,
            source_sha256=source_bound.sha256,
            analysis_sha256=analysis_bound.sha256,
        )
        supersedes = {
            "path": str(superseded_bound.path),
            "sha256": superseded_bound.sha256,
            "schema": superseded["schema"],
            "receipt_sha256": superseded["receipt_sha256"],
            "reason": _SUPERSESSION_REASON,
        }
    commit = receipt["active_source_identity"]["repo_git_commit"]
    body: dict[str, object] = {
        "schema": SCHEMA,
        "status": "verified_exact_recomputation",
        "kind": kind,
        "source": {
            "path": str(source_bound.path),
            "sha256": source_bound.sha256,
            "checkpoint_sha256": source["checkpoint_sha256"],
            "publication_identity_sha256": receipt["publication_identity_sha256"],
            "repo_git_commit": commit,
            "schema": receipt["schema"],
            "status": receipt["status"],
            "partial": receipt["partial"],
            "tensors_done": receipt["tensors_done"],
            "glm_rate_plan": receipt["glm_rate_plan"],
        },
        "analysis": {
            "path": str(analysis_bound.path),
            "sha256": analysis_bound.sha256,
            "schema": analysis["schema"],
        },
        "population_counts": receipt["population_counts"],
        "aggregation_contract": "dense/routed population-separated; no pooled median",
        "verifier": {
            "path": str(verifier_bound.path),
            "sha256": verifier_bound.sha256,
            "dependencies": [
                {"path": str(bound.path), "sha256": bound.sha256}
                for bound in dependency_bounds
            ],
        },
        "supersedes": supersedes,
    }
    return {**body, "receipt_sha256": _identity_sha256(body)}


def validate_receipt_self_digest(receipt: Mapping[str, object]) -> None:
    if set(receipt) != {
        "schema", "status", "kind", "source", "analysis",
        "population_counts", "aggregation_contract", "verifier",
        "supersedes", "receipt_sha256",
    }:
        raise AnalysisReceiptError("analysis receipt field set differs")
    digest = receipt.get("receipt_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise AnalysisReceiptError("analysis receipt digest is invalid")
    body = {key: value for key, value in receipt.items()
            if key != "receipt_sha256"}
    if digest != _identity_sha256(body):
        raise AnalysisReceiptError("analysis receipt self-digest differs")
    kind = receipt.get("kind")
    expected_plan = _EXPECTED_PLAN.get(kind)
    source = receipt.get("source")
    analysis = receipt.get("analysis")
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "verified_exact_recomputation"
        or expected_plan is None
        or not isinstance(source, dict)
        or source.get("glm_rate_plan") != expected_plan
        or not isinstance(analysis, dict)
        or analysis.get("schema")
        != (
            "trellis.coding_gain_table.v2"
            if kind == "coding_gain" else NEAR_FOUR_SCHEMA
        )
    ):
        raise AnalysisReceiptError("analysis receipt semantic identity differs")
    for owner, field in (
        (source, "sha256"),
        (source, "checkpoint_sha256"),
        (source, "publication_identity_sha256"),
        (analysis, "sha256"),
    ):
        value = owner.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise AnalysisReceiptError(
                f"analysis receipt {field} is invalid"
            )
    supersedes = receipt.get("supersedes")
    if supersedes is not None:
        if not isinstance(supersedes, dict) or set(supersedes) != {
            "path", "sha256", "schema", "receipt_sha256", "reason"
        }:
            raise AnalysisReceiptError("analysis receipt supersedes field differs")
        if (
            supersedes.get("schema")
            != "trellis.glm_e2m1_analysis_receipt.v1"
            or supersedes.get("reason") != _SUPERSESSION_REASON
        ):
            raise AnalysisReceiptError("analysis receipt supersession differs")
        for field in ("sha256", "receipt_sha256"):
            value = supersedes.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise AnalysisReceiptError(
                    f"analysis receipt supersedes {field} is invalid"
                )


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
    parser.add_argument("--supersedes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_receipt(
        kind=args.kind,
        source_path=args.source,
        analysis_path=args.analysis,
        supersedes_path=args.supersedes,
    )
    publish_receipt(args.out, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
