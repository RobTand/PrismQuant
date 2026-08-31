#!/usr/bin/env python3
"""Independently verify and seal a final GLM FP8-CB/TCQ result.

The campaign result remains the authority.  This read-only verifier checks its
closed receipt, recomputes every population-separated exact-byte frontier and
publishes one no-replace analysis receipt.  It also records descriptive
frontier-crossing diagnostics so a quality advantage is never mislabeled as
Pareto dominance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import stat
from typing import Mapping, Sequence

import atomic_publication as _ATOMIC_PUBLICATION
from atomic_publication import (
    PublicationError,
    atomic_checkpoint_json,
    canonical_json_bytes,
    exclusive_publication_claim,
    publish_file_no_replace,
)


SOURCE_SCHEMA = "trellis.glm_fp8_cb_tcq_two_bracket.v1"
SOURCE_STATUS = "measurement_complete_no_serving_verdict"
SCHEMA = "trellis.glm_fp8_cb_tcq_analysis_receipt.v1"
RATES = (4, 5)
CELL_MAP = {4: 32, 5: 40}
BRACKETS = ("production_row_fp32", "two_tier")
SELECTORS = ("lloyd", "exact_dp")
BOOK_PRICES = ("wire8", "fp16_production")
CLAIM_BOUNDARY = {
    "currency": "activation_importance_weighted_weight_sse",
    "activation_contract": "W*A16 screen; no activation quantizer executed",
    "population_aggregation": "dense and routed are separate; pooling forbidden",
    "serving_verdict": False,
    "runtime_claim": False,
    "performance_claim": False,
    "promotion_eligible": False,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MAX_BOUND_BYTES = 64 << 20


class AnalysisReceiptError(RuntimeError):
    pass


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_mode),
        int(info.st_uid), int(info.st_gid), int(info.st_size),
        int(info.st_mtime_ns), int(info.st_ctime_ns), int(info.st_nlink),
    )


def _read_bound_file(path: Path) -> tuple[bytes, dict[str, object]]:
    """Read one stable regular-file inode and bind its exact bytes once."""

    candidate = path.absolute()
    if not hasattr(os, "O_NOFOLLOW"):
        raise AnalysisReceiptError("O_NOFOLLOW is required for bound inputs")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise AnalysisReceiptError(f"cannot open bound file {candidate}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AnalysisReceiptError(f"bound input is not regular: {candidate}")
        if before.st_size < 0 or before.st_size > _MAX_BOUND_BYTES:
            raise AnalysisReceiptError(
                f"bound input exceeds {_MAX_BOUND_BYTES} bytes: {candidate}"
            )
        chunks = []
        total = 0
        while chunk := os.read(descriptor, min(8 << 20, _MAX_BOUND_BYTES + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_BOUND_BYTES:
                raise AnalysisReceiptError(
                    f"bound input exceeds {_MAX_BOUND_BYTES} bytes: {candidate}"
                )
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(candidate, follow_symlinks=False)
        except OSError as exc:
            raise AnalysisReceiptError(
                f"bound input path changed while read: {candidate}"
            ) from exc
        identity = _file_identity(before)
        if (
            _file_identity(after) != identity
            or _file_identity(path_after) != identity
            or not stat.S_ISREG(path_after.st_mode)
        ):
            raise AnalysisReceiptError(
                f"bound input identity changed while read: {candidate}"
            )
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise AnalysisReceiptError(f"bound input size changed while read: {candidate}")
        return raw, {
            "path": str(candidate),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    finally:
        os.close(descriptor)


def _stable_file_sha256(path: Path) -> str:
    return str(_read_bound_file(path)[1]["sha256"])


_VERIFIER_PATH = Path(__file__).absolute()
_DEPENDENCY_PATH = Path(_ATOMIC_PUBLICATION.__file__).absolute()
_IMPORT_VERIFIER_BINDING = _read_bound_file(_VERIFIER_PATH)[1]
_IMPORT_DEPENDENCY_BINDING = _read_bound_file(_DEPENDENCY_PATH)[1]


def _strict_json_object(
    path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str):
        raise ValueError(f"non-finite JSON constant {token!r}")

    raw, binding = _read_bound_file(path)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError) as exc:
        raise AnalysisReceiptError(f"invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisReceiptError(f"{path} is not one JSON object")
    return value, binding


def _identity_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite(value: object, *, where: str, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AnalysisReceiptError(f"{where} must be finite")
    result = float(value)
    if nonnegative and result < 0:
        raise AnalysisReceiptError(f"{where} must be nonnegative")
    return result


def _arm_names() -> set[str]:
    cb = {
        f"fp8_cb_{kind}@{rung}"
        for kind in ("fixed", "learned")
        for rung in CELL_MAP.values()
    }
    tcq = {
        f"tcq_e4m3.{bracket}.{selector}@{rate}"
        for bracket in BRACKETS
        for selector in SELECTORS
        for rate in RATES
    }
    return cb | tcq


ARM_NAMES = _arm_names()


def _bpw(arm: Mapping[str, object], *, book_price: str) -> float:
    footprint = arm.get("footprint")
    if not isinstance(footprint, Mapping):
        raise AnalysisReceiptError("arm footprint is absent")
    field = (
        "exact_bpw_book_wire8"
        if arm.get("book_kind") == "per_tensor_weighted_lloyd"
        and book_price == "wire8"
        else "exact_bpw"
    )
    return _finite(footprint.get(field), where=f"footprint.{field}")


def _frontier(points: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for point in points:
        dominated = any(
            other is not point
            and other["bpw"] <= point["bpw"]
            and other["snr"] >= point["snr"]
            and (other["bpw"] < point["bpw"] or other["snr"] > point["snr"])
            for other in points
        )
        if not dominated:
            result.append(point)
    return sorted(result, key=lambda item: (item["bpw"], -item["snr"], item["arm"]))


def _family_dominates(
    candidate: Sequence[dict[str, object]], other: Sequence[dict[str, object]]
) -> bool:
    return all(
        any(
            left["bpw"] <= right["bpw"]
            and left["snr"] >= right["snr"]
            and (left["bpw"] < right["bpw"] or left["snr"] > right["snr"])
            for left in candidate
        )
        for right in other
    )


def _tensor_verdict(
    cell: Mapping[str, object], *, rate: int, bracket: str, book_price: str
) -> dict[str, object]:
    arms = cell["arms"]
    rung = CELL_MAP[rate]
    cb = _frontier([
        {
            "arm": name,
            "bpw": _bpw(arms[name], book_price=book_price),
            "snr": float(arms[name]["weighted_snr_db"]),
        }
        for name in (f"fp8_cb_fixed@{rung}", f"fp8_cb_learned@{rung}")
    ])
    tcq = _frontier([
        {
            "arm": name,
            "bpw": _bpw(arms[name], book_price=book_price),
            "snr": float(arms[name]["weighted_snr_db"]),
        }
        for name in (
            f"tcq_e4m3.{bracket}.lloyd@{rate}",
            f"tcq_e4m3.{bracket}.exact_dp@{rate}",
        )
    ])
    cb_dominates = _family_dominates(cb, tcq)
    tcq_dominates = _family_dominates(tcq, cb)
    if cb_dominates and tcq_dominates:
        raise AnalysisReceiptError("mutual strict family dominance is impossible")
    best_cb = max(cb, key=lambda point: point["snr"])
    best_tcq = max(tcq, key=lambda point: point["snr"])
    return {
        "verdict": (
            "FP8_CB" if cb_dominates else
            "TCQ_E4M3" if tcq_dominates else
            "NO_VERDICT_exact_byte_frontiers_cross"
        ),
        "cb_frontier": cb,
        "tcq_frontier": tcq,
        "best_quality_cb_minus_tcq_db": best_cb["snr"] - best_tcq["snr"],
        "best_quality_cb_bpw": best_cb["bpw"],
        "best_quality_tcq_bpw": best_tcq["bpw"],
        "minimum_cb_bpw": min(point["bpw"] for point in cb),
        "minimum_tcq_bpw": min(point["bpw"] for point in tcq),
    }


def _population_summaries(
    per_tensor: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for population in ("dense", "routed"):
        cells = [
            cell for cell in per_tensor.values()
            if cell.get("population") == population
        ]
        bracket_rows = []
        final_cells = []
        for rate in RATES:
            combination_verdicts = []
            for bracket in BRACKETS:
                for book_price in BOOK_PRICES:
                    rows = [
                        _tensor_verdict(
                            cell, rate=rate, bracket=bracket,
                            book_price=book_price,
                        )
                        for cell in cells
                    ]
                    counts = {
                        label: sum(row["verdict"] == label for row in rows)
                        for label in (
                            "FP8_CB", "TCQ_E4M3",
                            "NO_VERDICT_exact_byte_frontiers_cross",
                        )
                    }
                    unanimous = (
                        "FP8_CB" if counts["FP8_CB"] == len(rows) else
                        "TCQ_E4M3" if counts["TCQ_E4M3"] == len(rows) else
                        "NO_VERDICT_mixed_or_crossing"
                    )
                    combination_verdicts.append(unanimous)
                    bracket_rows.append({
                        "nominal_body_bpw": rate,
                        "fp8_cb_rung": CELL_MAP[rate],
                        "trellis_scale_bracket": bracket,
                        "learned_book_price_bracket": book_price,
                        "tensors": len(rows),
                        "counts": counts,
                        "verdict": unanimous,
                        "best_quality_cb_minus_tcq_db_median": statistics.median(
                            row["best_quality_cb_minus_tcq_db"] for row in rows
                        ),
                        "best_quality_cb_bpw_median": statistics.median(
                            row["best_quality_cb_bpw"] for row in rows
                        ),
                        "best_quality_tcq_bpw_median": statistics.median(
                            row["best_quality_tcq_bpw"] for row in rows
                        ),
                    })
            final_cells.append({
                "nominal_body_bpw": rate,
                "fp8_cb_rung": CELL_MAP[rate],
                "verdict": (
                    combination_verdicts[0]
                    if len(set(combination_verdicts)) == 1
                    and combination_verdicts[0] in {"FP8_CB", "TCQ_E4M3"}
                    else "NO_VERDICT_brackets_disagree_or_frontiers_cross"
                ),
                "required_combination_verdicts": combination_verdicts,
            })
        output[population] = {
            "tensors": len(cells),
            "bracket_rows": bracket_rows,
            "cells": final_cells,
        }
    return output


def _frontier_diagnostics(
    per_tensor: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for population in ("dense", "routed"):
        cells = [
            cell for cell in per_tensor.values()
            if cell.get("population") == population
        ]
        rows = []
        for rate in RATES:
            for bracket in BRACKETS:
                for book_price in BOOK_PRICES:
                    values = [
                        _tensor_verdict(
                            cell, rate=rate, bracket=bracket,
                            book_price=book_price,
                        )
                        for cell in cells
                    ]
                    rows.append({
                        "nominal_body_bpw": rate,
                        "trellis_scale_bracket": bracket,
                        "learned_book_price_bracket": book_price,
                        "tensors": len(values),
                        "tcq_best_quality_higher": sum(
                            value["best_quality_cb_minus_tcq_db"] < 0
                            for value in values
                        ),
                        "cb_minimum_bpw_lower": sum(
                            value["minimum_cb_bpw"] < value["minimum_tcq_bpw"]
                            for value in values
                        ),
                        "minimum_cb_minus_tcq_bpw_median": statistics.median(
                            value["minimum_cb_bpw"] - value["minimum_tcq_bpw"]
                            for value in values
                        ),
                    })
        output[population] = rows
    return output


def _validate_source(source: Mapping[str, object]) -> Mapping[str, object]:
    expected = {
        "schema", "settings", "started_at_unix_s", "per_tensor", "partial",
        "tensors_done", "execution_segments", "checkpoint_sha256",
        "completed_at_unix_s", "population_summaries", "status",
        "claim_boundary",
    }
    if set(source) != expected:
        raise AnalysisReceiptError("source result field set differs")
    body = {key: value for key, value in source.items() if key != "checkpoint_sha256"}
    if source.get("checkpoint_sha256") != _identity_sha256(body):
        raise AnalysisReceiptError("source checkpoint self-digest differs")
    if (
        source.get("schema") != SOURCE_SCHEMA
        or source.get("status") != SOURCE_STATUS
        or source.get("partial") is not False
        or source.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise AnalysisReceiptError("source is not a final no-serving CB/TCQ result")
    settings = source.get("settings")
    per_tensor = source.get("per_tensor")
    if not isinstance(settings, Mapping) or not isinstance(per_tensor, Mapping):
        raise AnalysisReceiptError("source settings/per_tensor are malformed")
    if source.get("tensors_done") != len(per_tensor) or len(per_tensor) != 33:
        raise AnalysisReceiptError("source does not cover exactly 33 tensors")
    counts = {
        population: sum(cell.get("population") == population for cell in per_tensor.values())
        for population in ("dense", "routed")
    }
    if counts != {"dense": 9, "routed": 24}:
        raise AnalysisReceiptError("source population counts differ")
    if settings.get("population_counts") != counts:
        raise AnalysisReceiptError("settings population counts differ")
    if (
        settings.get("schema") != SOURCE_SCHEMA
        or settings.get("rungs") != [32, 40]
        or settings.get("rates") != [4.0, 5.0]
        or settings.get("cell_map") != {"4": 32, "5": 40}
        or settings.get("trellis_scale_brackets") != list(BRACKETS)
        or settings.get("alphabet_selectors") != list(SELECTORS)
        or settings.get("book_price_brackets") != list(BOOK_PRICES)
        or settings.get("encode_tier") != "balanced"
        or settings.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise AnalysisReceiptError("source campaign settings differ")
    environment = settings.get("environment")
    active = settings.get("active_source_identity")
    if not isinstance(environment, Mapping) or not isinstance(active, Mapping):
        raise AnalysisReceiptError("source execution identity is absent")
    commit = environment.get("repo_git_commit")
    if (
        not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None
        or active.get("repo_git_commit") != commit
        or environment.get("repo_tree_clean") is not True
        or environment.get("physical_host") != "sparky"
    ):
        raise AnalysisReceiptError("source execution identity differs")
    unsigned_settings = {
        key: value for key, value in settings.items()
        if key != "identity_sha256"
    }
    if settings.get("identity_sha256") != _identity_sha256(unsigned_settings):
        raise AnalysisReceiptError("source settings self-digest differs")
    files = active.get("files")
    if not isinstance(files, Mapping) or not files:
        raise AnalysisReceiptError("source active-file identity is absent")
    for label, item in files.items():
        if not isinstance(item, Mapping):
            raise AnalysisReceiptError(f"active source {label} is malformed")
        path = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(path, str) or not path.startswith("/")
            or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
        ):
            raise AnalysisReceiptError(f"active source {label} identity is invalid")
        if _stable_file_sha256(Path(path)) != digest:
            raise AnalysisReceiptError(f"active source {label} bytes differ")
    segments = source.get("execution_segments")
    if not isinstance(segments, list) or not segments:
        raise AnalysisReceiptError("source has no execution segment")
    segment_keys = {
        "schema", "physical_host", "container_id", "image_id", "gpu_uuid",
        "launch_attestation_path", "launch_attestation_sha256",
        "launch_command_sha256", "segment_sha256",
    }
    for segment in segments:
        if not isinstance(segment, Mapping) or set(segment) != segment_keys:
            raise AnalysisReceiptError("source execution segment field set differs")
        unsigned_segment = {
            key: value for key, value in segment.items()
            if key != "segment_sha256"
        }
        if (
            segment.get("schema") != "trellis.numeric_execution_segment.v1"
            or segment.get("physical_host") != environment.get("physical_host")
            or segment.get("image_id") != environment.get("container_image_id")
            or segment.get("gpu_uuid") != environment.get("gpu_uuid")
            or segment.get("segment_sha256") != _identity_sha256(unsigned_segment)
        ):
            raise AnalysisReceiptError("source execution segment identity differs")
        attestation_path = segment.get("launch_attestation_path")
        attestation_sha = segment.get("launch_attestation_sha256")
        if (
            not isinstance(attestation_path, str)
            or not isinstance(attestation_sha, str)
            or _stable_file_sha256(Path(attestation_path)) != attestation_sha
        ):
            raise AnalysisReceiptError("source launch attestation bytes differ")
    for name, cell in per_tensor.items():
        if not isinstance(cell, Mapping) or set(cell.get("arms", {})) != ARM_NAMES:
            raise AnalysisReceiptError(f"{name}: arm domain differs")
        rows, columns = cell.get("shape", (0, 0))
        numel = int(rows) * int(columns)
        energy = _finite(cell.get("weighted_energy"), where=f"{name}.energy")
        if numel <= 0 or energy <= 0:
            raise AnalysisReceiptError(f"{name}: invalid shape/energy")
        for arm_name, arm in cell["arms"].items():
            error = _finite(
                arm.get("weighted_sse"), where=f"{name}.{arm_name}.sse",
                nonnegative=True,
            )
            nsse = _finite(arm.get("weighted_nsse"), where=f"{name}.{arm_name}.nsse")
            snr = _finite(arm.get("weighted_snr_db"), where=f"{name}.{arm_name}.snr")
            if not math.isclose(nsse, error / energy, rel_tol=1e-12, abs_tol=1e-15):
                raise AnalysisReceiptError(f"{name}.{arm_name}: NSSE differs")
            if not math.isclose(
                snr, -10 * math.log10(max(nsse, 1e-300)),
                rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise AnalysisReceiptError(f"{name}.{arm_name}: SNR differs")
            footprint = arm.get("footprint")
            if not isinstance(footprint, Mapping):
                raise AnalysisReceiptError(f"{name}.{arm_name}: footprint absent")
            exact = _finite(footprint.get("exact_bpw"), where="exact_bpw")
            expected_bpw = (
                int(footprint["total_bits"]) / numel
                if "total_bits" in footprint
                else int(footprint["total_bytes"]) * 8 / numel
            )
            if not math.isclose(exact, expected_bpw, rel_tol=0, abs_tol=1e-12):
                raise AnalysisReceiptError(f"{name}.{arm_name}: exact bpw differs")
            if arm.get("book_kind") == "per_tensor_weighted_lloyd":
                wire = _finite(
                    footprint.get("exact_bpw_book_wire8"),
                    where="exact_bpw_book_wire8",
                )
                wire_expected = (
                    int(footprint["total_bits"])
                    - int(footprint["codebook_side_bits"])
                    + int(footprint["codebook_side_bits_wire8"])
                ) / numel
                if not math.isclose(wire, wire_expected, rel_tol=0, abs_tol=1e-12):
                    raise AnalysisReceiptError(f"{name}.{arm_name}: wire8 bpw differs")
    summaries = _population_summaries(per_tensor)
    if source.get("population_summaries") != summaries:
        raise AnalysisReceiptError("source population summaries differ on recomputation")
    return settings


def build_receipt(source_path: Path) -> dict[str, object]:
    verifier_binding = _read_bound_file(_VERIFIER_PATH)[1]
    dependency_binding = _read_bound_file(_DEPENDENCY_PATH)[1]
    if verifier_binding != _IMPORT_VERIFIER_BINDING:
        raise AnalysisReceiptError("verifier bytes changed after module import")
    if dependency_binding != _IMPORT_DEPENDENCY_BINDING:
        raise AnalysisReceiptError("publication dependency changed after module import")
    source, source_binding = _strict_json_object(source_path)
    settings = _validate_source(source)
    per_tensor = source["per_tensor"]
    body: dict[str, object] = {
        "schema": SCHEMA,
        "status": "verified_exact_recomputation",
        "source": {
            "path": source_binding["path"],
            "sha256": source_binding["sha256"],
            "size_bytes": source_binding["size_bytes"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "repo_git_commit": settings["environment"]["repo_git_commit"],
            "schema": source["schema"],
            "status": source["status"],
            "partial": source["partial"],
            "tensors_done": source["tensors_done"],
            "settings_identity_sha256": settings["identity_sha256"],
        },
        "population_counts": settings["population_counts"],
        "aggregation_contract": "dense/routed population-separated; no pooled field",
        "verdict_contract": (
            "strict exact-byte family coverage on every tensor under both scale "
            "and learned-book price brackets; crossing is NO_VERDICT"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "population_summaries": _population_summaries(per_tensor),
        "frontier_diagnostics": _frontier_diagnostics(per_tensor),
        "verifier": {
            "path": verifier_binding["path"],
            "sha256": verifier_binding["sha256"],
            "size_bytes": verifier_binding["size_bytes"],
            "dependencies": [{
                "path": dependency_binding["path"],
                "sha256": dependency_binding["sha256"],
                "size_bytes": dependency_binding["size_bytes"],
            }],
        },
    }
    return {**body, "receipt_sha256": _identity_sha256(body)}


def validate_receipt(receipt: Mapping[str, object]) -> None:
    digest = receipt.get("receipt_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise AnalysisReceiptError("analysis receipt digest is invalid")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if digest != _identity_sha256(body):
        raise AnalysisReceiptError("analysis receipt self-digest differs")


def publish_receipt(output: Path, receipt: Mapping[str, object]) -> None:
    validate_receipt(receipt)
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_receipt(args.source)
    publish_receipt(args.out, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
