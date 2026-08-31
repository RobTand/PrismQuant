#!/usr/bin/env python3
"""Hardened two-bracket FP8-CB vs E4M3-TCQ screen on GLM BF16 weights.

This research-only driver closes one narrowly defined numeric gap: the 4.0 and
5.0 nominal-body-bit cells that the 2026-08-30 handover left unresolved.  It
uses the immutable finalized GLM corpus and the frozen 2026-08-29 FP8 ladder
implementations.  One result contains both E4M3 scale-plane brackets, both
audited alphabet selectors, and balanced fixed/learned FP8-CB controls.

The currency is activation-importance-weighted weight SSE.  Exact serialized
bpw is retained for every point, including row scales, trellis metadata, and
both honest learned-book prices.  Dense and routed populations are never
pooled.  The output is a codec screen, not a W8A8, KL/PPL, runtime, or serving
verdict.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

import fp8_learned_glm as BASE  # noqa: E402
from atomic_publication import (  # noqa: E402
    PublicationError,
    atomic_checkpoint_json,
    exclusive_publication_claim,
    file_sha256,
    identity_sha256,
    publish_file_no_replace,
)
from isolated_glm_corpus import load_active_glm_corpus_bound  # noqa: E402
from numeric_execution_contract import (  # noqa: E402
    NumericExecutionContractError,
    require_numeric_execution_environment,
    require_repo_commit,
    validate_numeric_execution_record,
)


SCHEMA = "trellis.glm_fp8_cb_tcq_two_bracket.v1"
STATUS = "measurement_complete_no_serving_verdict"
CLAIM_BOUNDARY = {
    "currency": "activation_importance_weighted_weight_sse",
    "activation_contract": "W*A16 screen; no activation quantizer executed",
    "population_aggregation": "dense and routed are separate; pooling forbidden",
    "serving_verdict": False,
    "runtime_claim": False,
    "performance_claim": False,
    "promotion_eligible": False,
}
RUNGS = (32, 40)
RATES = (4.0, 5.0)
CELL_MAP = {4: 32, 5: 40}
TRELLIS_BRACKETS = ("production_row_fp32", "two_tier")
ALPHABET_SELECTORS = ("lloyd", "exact_dp")
BOOK_PRICE_BRACKETS = ("wire8", "fp16_production")
ENCODE_TIER = "balanced"
DEFAULT_LOCKED_LADDER = BASE.DEFAULT_LOCKED_LADDER
EXPECTED_DP_SHA256 = "022cd576c052cf613eb856a8ad4fce94462e819cb23274815e297f0493491696"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SETTINGS_KEYS = {
    "schema", "corpus_manifest", "corpus_manifest_sha256",
    "corpus_file_sha256", "importance_value_sha256",
    "corpus_prismaquant_commit", "population_counts", "rungs", "rates",
    "cell_map", "trellis_scale_brackets", "alphabet_selectors",
    "book_price_brackets", "encode_tier", "locked_sources",
    "frozen_codec_closure", "active_source_identity", "environment",
    "command", "claim_boundary", "identity_sha256",
}


class CampaignError(RuntimeError):
    """The campaign cannot make an unambiguous numeric claim."""


def _exact_keys(value: Mapping[str, object], expected: set[str], *, where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CampaignError(
            f"{where} members differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _repo_commit() -> str:
    try:
        return require_repo_commit(REPO_ROOT)
    except NumericExecutionContractError as exc:
        raise CampaignError(str(exc)) from exc


def _active_source_identity() -> dict[str, object]:
    files = {
        "driver": Path(__file__).resolve(strict=True),
        "base_fp8_glm_driver": HERE / "fp8_learned_glm.py",
        "atomic_publication": HERE / "atomic_publication.py",
        "isolated_glm_corpus": HERE / "isolated_glm_corpus.py",
        "numeric_execution_contract": HERE / "numeric_execution_contract.py",
        "active_corpus_reader": REPO_ROOT / "prismaquant/trellis_bf16_corpus.py",
    }
    return {
        "repo_root": str(REPO_ROOT),
        "repo_git_commit": _repo_commit(),
        "files": {
            name: {
                "path": str(path.resolve(strict=True)),
                "sha256": file_sha256(path.resolve(strict=True)),
            }
            for name, path in sorted(files.items())
        },
    }


def _locked_sources(ladder_path: Path) -> dict[str, object]:
    locked = BASE._locked_sources(ladder_path)
    dp = (
        ladder_path.resolve(strict=True).parent.parent
        / "trellis-stage0/e4m3_alphabet_dp.py"
    ).resolve(strict=True)
    if file_sha256(dp) != EXPECTED_DP_SHA256:
        raise CampaignError("locked exact-DP alphabet source hash differs")
    return {
        **locked,
        "e4m3_alphabet_dp_path": str(dp),
        "e4m3_alphabet_dp_sha256": EXPECTED_DP_SHA256,
    }


def _execution_environment(ladder) -> dict[str, object]:
    try:
        return require_numeric_execution_environment(
            REPO_ROOT,
            ladder.H.current_env(),
            os.environ,
            require_cuda=True,
        )
    except NumericExecutionContractError as exc:
        raise CampaignError(str(exc)) from exc


def _settings(*, args, corpus, manifest_binding, ladder, execution) -> dict[str, object]:
    settings: dict[str, object] = {
        "schema": SCHEMA,
        "corpus_manifest": str(corpus.manifest_path),
        "corpus_manifest_sha256": manifest_binding["sha256"],
        "corpus_file_sha256": corpus.manifest["file_sha256"],
        "importance_value_sha256": corpus.manifest["importance_identity"][
            "value_sha256"
        ],
        "corpus_prismaquant_commit": corpus.manifest["prismaquant_commit"],
        "population_counts": {
            name: len(entries) for name, entries in corpus.populations.items()
        },
        "rungs": list(RUNGS),
        "rates": list(RATES),
        "cell_map": {str(rate): rung for rate, rung in CELL_MAP.items()},
        "trellis_scale_brackets": list(TRELLIS_BRACKETS),
        "alphabet_selectors": list(ALPHABET_SELECTORS),
        "book_price_brackets": list(BOOK_PRICE_BRACKETS),
        "encode_tier": ENCODE_TIER,
        "locked_sources": _locked_sources(args.locked_ladder),
        "frozen_codec_closure": BASE._frozen_codec_closure(ladder),
        "active_source_identity": _active_source_identity(),
        "environment": execution,
        "command": list(sys.argv),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    settings["identity_sha256"] = identity_sha256(settings)
    _validate_settings(settings, corpus.entries)
    return settings


@contextmanager
def _alphabet_mode(mode: str):
    if mode not in ALPHABET_SELECTORS:
        raise CampaignError(f"unsupported alphabet selector {mode!r}")
    previous = os.environ.get("E4M3_ALPHABET")
    os.environ["E4M3_ALPHABET"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("E4M3_ALPHABET", None)
        else:
            os.environ["E4M3_ALPHABET"] = previous


def _arm_names() -> tuple[str, ...]:
    cb = [
        f"fp8_cb_fixed@{rung}" for rung in RUNGS
    ] + [f"fp8_cb_learned@{rung}" for rung in RUNGS]
    tcq = [
        f"tcq_e4m3.{bracket}.{selector}@{int(rate)}"
        for bracket in TRELLIS_BRACKETS
        for selector in ALPHABET_SELECTORS
        for rate in RATES
    ]
    return tuple((*cb, *tcq))


ARM_NAMES = _arm_names()
_ARM_COMMON_KEYS = {
    "encode_seconds_observation_not_perf_claim",
    "weighted_sse",
    "weighted_nsse",
    "weighted_snr_db",
    "reconstruction_sha256",
    "footprint",
    "family",
}
_CB_ARM_KEYS = _ARM_COMMON_KEYS | {"rung", "encode_tier", "book_kind"}
_CB_LEARNED_ARM_KEYS = _CB_ARM_KEYS | {"learned_book"}
_TCQ_ARM_KEYS = _ARM_COMMON_KEYS | {
    "rate",
    "trellis_scale_bracket",
    "alphabet_selector",
    "e4m3_plane_sha256",
    "alphabet",
    "schedule",
}


def _metric_row(
    *, ladder, weight, reconstruction, metric, energy: float, footprint,
    elapsed: float, metadata: Mapping[str, object],
) -> dict[str, object]:
    error = float(ladder.C.weighted_sse(weight, reconstruction, metric))
    if not math.isfinite(error) or error < 0:
        raise CampaignError("encoder returned a non-finite weighted error")
    nsse = error / energy
    return {
        "encode_seconds_observation_not_perf_claim": float(elapsed),
        "weighted_sse": error,
        "weighted_nsse": nsse,
        "weighted_snr_db": -10.0 * math.log10(max(nsse, 1e-300)),
        "reconstruction_sha256": ladder.tensor_sha256(
            reconstruction.contiguous()
        ),
        "footprint": footprint,
        **dict(metadata),
    }


def _measure_cell(*, entry, corpus, ladder, device) -> dict[str, object]:
    import torch

    raw, importance = corpus.load_tensor(entry)
    weight = raw.to(device=device, dtype=torch.float32)
    importance_device = importance.to(device=device, dtype=torch.float32)
    rows, columns = map(int, weight.shape)
    eff = ladder.P.eff_scale_plane(weight)
    ladder.assert_context_parity(weight, importance_device, eff)
    _x, _pes, metric, _enc = ladder.context_from_plane(
        weight, importance_device, eff
    )
    energy = float(ladder.C.weighted_sse(weight, torch.zeros_like(weight), metric))
    if not math.isfinite(energy) or energy <= 0:
        raise CampaignError(f"{entry.name}: invalid weighted energy")
    cell: dict[str, object] = {
        "population": entry.population,
        "shape": [rows, columns],
        "source_weight_sha256": entry.source_weight_sha256,
        "importance_sha256": entry.importance_sha256,
        "importance_source": {
            "qname": entry.importance_source_qname,
            "expert": entry.importance_source_expert,
            "denominator_name": entry.importance_denominator_name,
            "denominator": entry.importance_denominator,
        },
        "metric_weight_sha256": ladder.tensor_sha256(metric.contiguous()),
        "weighted_energy": energy,
        "arms": {},
    }
    arms = cell["arms"]
    assert isinstance(arms, dict)

    for rung in RUNGS:
        fixed, elapsed = ladder.timed(
            lambda k=rung: ladder.cb_arm_fp8(
                weight, k, metric, encode_tier=ENCODE_TIER
            ),
            sync=True,
        )
        name = f"fp8_cb_fixed@{rung}"
        arms[name] = _metric_row(
            ladder=ladder,
            weight=weight,
            reconstruction=fixed,
            metric=metric,
            energy=energy,
            footprint=ladder.fp8_cb_payload([rows, columns], rung),
            elapsed=elapsed,
            metadata={
                "family": "FP8_CB_K",
                "rung": rung,
                "encode_tier": ENCODE_TIER,
                "book_kind": "fixed_lattice",
            },
        )
        (learned, tables), elapsed = ladder.timed(
            lambda k=rung: ladder.cb_arm_fp8_learned(
                weight, k, metric, encode_tier=ENCODE_TIER
            ),
            sync=True,
        )
        book = ladder.assert_legal_e4m3_book(tables, rung)
        name = f"fp8_cb_learned@{rung}"
        arms[name] = _metric_row(
            ladder=ladder,
            weight=weight,
            reconstruction=learned,
            metric=metric,
            energy=energy,
            footprint=ladder.fp8_cb_payload(
                [rows, columns],
                rung,
                learned=True,
                book_shapes=[tuple(table.shape) for table in tables],
            ),
            elapsed=elapsed,
            metadata={
                "family": "FP8_CB_K",
                "rung": rung,
                "encode_tier": ENCODE_TIER,
                "book_kind": "per_tensor_weighted_lloyd",
                "learned_book": book,
            },
        )

    for bracket in TRELLIS_BRACKETS:
        plane = (
            ladder.row_fp32_scale_plane(weight)
            if bracket == ladder.PRODUCTION_SCALE
            else eff
        )
        x, pes, _metric_again, enc_weight = ladder.context_from_plane(
            weight, importance_device, plane
        )
        col_weight = ladder.S4.column_weight(enc_weight)
        for selector in ALPHABET_SELECTORS:
            with _alphabet_mode(selector):
                codes, _scalar, alphabet = ladder.e4m3_alphabets(x, enc_weight)
            if alphabet.get("alphabet_mode") != selector:
                raise CampaignError("frozen alphabet selector did not honor its mode")
            for rate in RATES:
                schedule, schedule_meta = ladder.C.rwf_schedule(
                    rate, col_weight, maximum_rate=8
                )
                schedule_list = [int(value) for value in schedule.tolist()]
                used = ladder.used_alphabets_e4m3(schedule, codes)
                footprint = ladder.e4m3_trellis_payload(
                    [rows, columns],
                    int(round(rate * ladder.SUPERBLOCK)),
                    schedule_list,
                    used,
                    bracket,
                )
                reconstruction, elapsed = ladder.timed(
                    lambda: ladder.trellis_arm_e4m3(
                        x,
                        enc_weight,
                        pes,
                        codes,
                        schedule,
                        backend="triton",
                    ),
                    sync=True,
                )
                name = f"tcq_e4m3.{bracket}.{selector}@{int(rate)}"
                arms[name] = _metric_row(
                    ladder=ladder,
                    weight=weight,
                    reconstruction=reconstruction,
                    metric=metric,
                    energy=energy,
                    footprint=footprint,
                    elapsed=elapsed,
                    metadata={
                        "family": "TCQ_E4M3_R256",
                        "rate": rate,
                        "trellis_scale_bracket": bracket,
                        "alphabet_selector": selector,
                        "e4m3_plane_sha256": ladder.tensor_sha256(plane),
                        "alphabet": alphabet,
                        "schedule": {
                            **schedule_meta,
                            **ladder.C.schedule_statistics(schedule, 8),
                        },
                    },
                )
    if set(arms) != set(ARM_NAMES):
        raise CampaignError(f"{entry.name}: generated arm set is incomplete")
    return cell


def _bpw(arm: Mapping[str, object], *, book_price: str) -> float:
    footprint = arm["footprint"]
    if not isinstance(footprint, Mapping):
        raise CampaignError("arm footprint must be an object")
    if arm.get("book_kind") == "per_tensor_weighted_lloyd" and book_price == "wire8":
        value = footprint.get("exact_bpw_book_wire8")
    else:
        value = footprint.get("exact_bpw")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CampaignError("arm exact bpw is missing or non-finite")
    return float(value)


def _frontier(points: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            no_worse = other["bpw"] <= point["bpw"] and other["snr"] >= point["snr"]
            strict = other["bpw"] < point["bpw"] or other["snr"] > point["snr"]
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            result.append(point)
    return sorted(result, key=lambda point: (point["bpw"], -point["snr"], point["arm"]))


def _family_dominates(
    candidate: Sequence[dict[str, object]], other: Sequence[dict[str, object]]
) -> bool:
    """True only when the candidate frontier covers every opposing point."""
    return all(
        any(
            left["bpw"] <= right["bpw"]
            and left["snr"] >= right["snr"]
            and (left["bpw"] < right["bpw"] or left["snr"] > right["snr"])
            for left in candidate
        )
        for right in other
    )


def _tensor_cell_verdict(
    cell: Mapping[str, object], *, rate: int, bracket: str, book_price: str
) -> dict[str, object]:
    rung = CELL_MAP[rate]
    arms = cell["arms"]
    cb_names = (f"fp8_cb_fixed@{rung}", f"fp8_cb_learned@{rung}")
    tcq_names = tuple(
        f"tcq_e4m3.{bracket}.{selector}@{rate}"
        for selector in ALPHABET_SELECTORS
    )
    cb = _frontier([
        {
            "arm": name,
            "bpw": _bpw(arms[name], book_price=book_price),
            "snr": float(arms[name]["weighted_snr_db"]),
        }
        for name in cb_names
    ])
    tcq = _frontier([
        {
            "arm": name,
            "bpw": _bpw(arms[name], book_price=book_price),
            "snr": float(arms[name]["weighted_snr_db"]),
        }
        for name in tcq_names
    ])
    cb_dominates = _family_dominates(cb, tcq)
    tcq_dominates = _family_dominates(tcq, cb)
    if cb_dominates and tcq_dominates:
        raise CampaignError("mutual strict family dominance is impossible")
    verdict = (
        "FP8_CB"
        if cb_dominates
        else "TCQ_E4M3"
        if tcq_dominates
        else "NO_VERDICT_exact_byte_frontiers_cross"
    )
    best_cb = max(cb, key=lambda point: point["snr"])
    best_tcq = max(tcq, key=lambda point: point["snr"])
    return {
        "verdict": verdict,
        "cb_frontier": cb,
        "tcq_frontier": tcq,
        "best_quality_cb_minus_tcq_db": best_cb["snr"] - best_tcq["snr"],
        "best_quality_cb_bpw": best_cb["bpw"],
        "best_quality_tcq_bpw": best_tcq["bpw"],
    }


def population_summaries(per_tensor: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    groups: dict[str, list[tuple[str, Mapping[str, object]]]] = {
        "dense": [],
        "routed": [],
    }
    for name, cell in per_tensor.items():
        population = cell.get("population")
        if population not in groups:
            raise CampaignError(f"{name}: invalid population {population!r}")
        groups[population].append((name, cell))
    summaries: dict[str, object] = {}
    for population, named_cells in groups.items():
        bracket_rows = []
        final_cells = []
        for rate in CELL_MAP:
            combination_verdicts = []
            for bracket in TRELLIS_BRACKETS:
                for book_price in BOOK_PRICE_BRACKETS:
                    rows = [
                        _tensor_cell_verdict(
                            cell, rate=rate, bracket=bracket, book_price=book_price
                        )
                        for _name, cell in named_cells
                    ]
                    counts = {
                        label: sum(row["verdict"] == label for row in rows)
                        for label in (
                            "FP8_CB",
                            "TCQ_E4M3",
                            "NO_VERDICT_exact_byte_frontiers_cross",
                        )
                    }
                    unanimous = (
                        "FP8_CB"
                        if counts["FP8_CB"] == len(rows)
                        else "TCQ_E4M3"
                        if counts["TCQ_E4M3"] == len(rows)
                        else "NO_VERDICT_mixed_or_crossing"
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
                        "best_quality_cb_minus_tcq_db_median": median(
                            row["best_quality_cb_minus_tcq_db"] for row in rows
                        ),
                        "best_quality_cb_bpw_median": median(
                            row["best_quality_cb_bpw"] for row in rows
                        ),
                        "best_quality_tcq_bpw_median": median(
                            row["best_quality_tcq_bpw"] for row in rows
                        ),
                    })
            final_verdict = (
                combination_verdicts[0]
                if len(set(combination_verdicts)) == 1
                and combination_verdicts[0] in {"FP8_CB", "TCQ_E4M3"}
                else "NO_VERDICT_brackets_disagree_or_frontiers_cross"
            )
            final_cells.append({
                "nominal_body_bpw": rate,
                "fp8_cb_rung": CELL_MAP[rate],
                "verdict": final_verdict,
                "required_combination_verdicts": combination_verdicts,
            })
        summaries[population] = {
            "tensors": len(named_cells),
            "bracket_rows": bracket_rows,
            "cells": final_cells,
        }
    return summaries


def _replay_semantics(cell: Mapping[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(dict(cell))
    for arm in normalized.get("arms", {}).values():
        arm.pop("encode_seconds_observation_not_perf_claim", None)
    return normalized


def require_replay_match(
    name: str, saved: Mapping[str, object], regenerated: Mapping[str, object]
) -> None:
    if _replay_semantics(saved) != _replay_semantics(regenerated):
        raise CampaignError(
            f"{name}: saved cell differs from deterministic full replay"
        )


def _sealed(report: Mapping[str, object]) -> dict[str, object]:
    body = {key: value for key, value in report.items() if key != "checkpoint_sha256"}
    return {**body, "checkpoint_sha256": identity_sha256(body)}


def _strict_report(path: Path) -> dict[str, object]:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CampaignError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise CampaignError(f"invalid checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError("checkpoint must be one JSON object")
    return value


def _finite(value: object, *, where: str, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CampaignError(f"{where} must be finite")
    result = float(value)
    if nonnegative and result < 0:
        raise CampaignError(f"{where} must be nonnegative")
    return result


def _validate_settings(settings: Mapping[str, object], entries) -> None:
    _exact_keys(settings, _SETTINGS_KEYS, where="settings")
    if settings.get("schema") != SCHEMA:
        raise CampaignError("settings.schema differs")
    for field in (
        "corpus_manifest_sha256",
        "corpus_file_sha256",
        "importance_value_sha256",
    ):
        if not _SHA256_RE.fullmatch(str(settings.get(field))):
            raise CampaignError(f"settings.{field} is invalid")
    if not re.fullmatch(
        r"[0-9a-f]{40}", str(settings.get("corpus_prismaquant_commit"))
    ):
        raise CampaignError("settings.corpus_prismaquant_commit is invalid")
    expected_populations: dict[str, int] = {}
    for entry in entries:
        expected_populations[entry.population] = (
            expected_populations.get(entry.population, 0) + 1
        )
    if settings.get("population_counts") != expected_populations:
        raise CampaignError("settings.population_counts differs from corpus")
    exact_constants = {
        "rungs": list(RUNGS),
        "rates": list(RATES),
        "cell_map": {str(rate): rung for rate, rung in CELL_MAP.items()},
        "trellis_scale_brackets": list(TRELLIS_BRACKETS),
        "alphabet_selectors": list(ALPHABET_SELECTORS),
        "book_price_brackets": list(BOOK_PRICE_BRACKETS),
        "encode_tier": ENCODE_TIER,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for field, expected in exact_constants.items():
        if settings.get(field) != expected:
            raise CampaignError(f"settings.{field} differs")
    command = settings.get("command")
    if not isinstance(command, list) or not command or any(
        not isinstance(value, str) for value in command
    ):
        raise CampaignError("settings.command is malformed")
    active_sources = settings.get("active_source_identity")
    if not isinstance(active_sources, Mapping):
        raise CampaignError("settings.active_source_identity is malformed")
    try:
        validate_numeric_execution_record(settings.get("environment"))
    except NumericExecutionContractError as exc:
        raise CampaignError(f"settings.environment: {exc}") from exc
    environment = settings["environment"]
    if active_sources.get("repo_git_commit") != environment.get("repo_git_commit"):
        raise CampaignError(
            "settings environment commit differs from active source identity"
        )
    if active_sources.get("repo_root") != environment.get("repo_root"):
        raise CampaignError(
            "settings environment repository differs from active source identity"
        )
    unsigned = {
        key: value for key, value in settings.items() if key != "identity_sha256"
    }
    if settings.get("identity_sha256") != identity_sha256(unsigned):
        raise CampaignError("settings identity digest differs")


def _validate_footprint(arm: Mapping[str, object], *, numel: int, where: str) -> None:
    footprint = arm.get("footprint")
    if not isinstance(footprint, Mapping):
        raise CampaignError(f"{where}.footprint must be an object")
    exact = _finite(footprint.get("exact_bpw"), where=f"{where}.exact_bpw")
    if "total_bits" in footprint:
        expected = int(footprint["total_bits"]) / numel
    elif "total_bytes" in footprint:
        expected = int(footprint["total_bytes"]) * 8.0 / numel
    else:
        raise CampaignError(f"{where}.footprint has no exact total")
    if not math.isclose(exact, expected, rel_tol=0, abs_tol=1e-12):
        raise CampaignError(f"{where}.exact_bpw differs from exact bytes")
    if arm.get("book_kind") == "per_tensor_weighted_lloyd":
        wire = _finite(
            footprint.get("exact_bpw_book_wire8"),
            where=f"{where}.exact_bpw_book_wire8",
        )
        if wire > exact:
            raise CampaignError(f"{where}: wire8 book price exceeds FP16 price")


def validate_report(
    report: Mapping[str, object], *, settings: Mapping[str, object], entries,
    require_complete: bool = False,
) -> None:
    _validate_settings(settings, entries)
    partial = report.get("partial")
    expected_keys = {
        "schema", "settings", "started_at_unix_s", "per_tensor", "partial",
        "tensors_done", "checkpoint_sha256",
    }
    if partial is False:
        expected_keys |= {
            "completed_at_unix_s", "population_summaries", "status",
            "claim_boundary",
        }
    elif partial is not True:
        raise CampaignError("report.partial must be boolean")
    _exact_keys(report, expected_keys, where="report")
    if report.get("schema") != SCHEMA or report.get("settings") != settings:
        raise CampaignError("report identity differs from campaign settings")
    body = {key: value for key, value in report.items() if key != "checkpoint_sha256"}
    if report.get("checkpoint_sha256") != identity_sha256(body):
        raise CampaignError("checkpoint self-digest differs")
    per_tensor = report.get("per_tensor")
    if not isinstance(per_tensor, Mapping):
        raise CampaignError("per_tensor must be an object")
    expected_names = [entry.name for entry in entries]
    names = list(per_tensor)
    if names != expected_names[: len(names)]:
        raise CampaignError("checkpoint tensors are not an exact corpus prefix")
    if report.get("tensors_done") != len(names):
        raise CampaignError("tensors_done differs from checkpoint coverage")
    by_name = {entry.name: entry for entry in entries}
    for name, cell in per_tensor.items():
        if not isinstance(cell, Mapping):
            raise CampaignError(f"{name}: cell must be an object")
        _exact_keys(cell, {
            "population", "shape", "source_weight_sha256", "importance_sha256",
            "importance_source", "metric_weight_sha256", "weighted_energy", "arms",
        }, where=f"per_tensor.{name}")
        entry = by_name[name]
        if (
            cell.get("population") != entry.population
            or tuple(cell.get("shape", ())) != entry.source_weight_shape
            or cell.get("source_weight_sha256") != entry.source_weight_sha256
            or cell.get("importance_sha256") != entry.importance_sha256
        ):
            raise CampaignError(f"{name}: cell/corpus binding differs")
        importance_source = cell.get("importance_source")
        if not isinstance(importance_source, Mapping):
            raise CampaignError(f"{name}: importance source must be an object")
        _exact_keys(
            importance_source,
            {"qname", "expert", "denominator_name", "denominator"},
            where=f"per_tensor.{name}.importance_source",
        )
        expected_importance_source = {
            "qname": entry.importance_source_qname,
            "expert": entry.importance_source_expert,
            "denominator_name": entry.importance_denominator_name,
            "denominator": entry.importance_denominator,
        }
        if importance_source != expected_importance_source:
            raise CampaignError(f"{name}: importance provenance differs")
        if not _SHA256_RE.fullmatch(str(cell.get("metric_weight_sha256"))):
            raise CampaignError(f"{name}: metric hash is invalid")
        energy = _finite(
            cell.get("weighted_energy"), where=f"{name}.weighted_energy"
        )
        if energy <= 0:
            raise CampaignError(f"{name}: weighted energy must be positive")
        arms = cell.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(ARM_NAMES):
            raise CampaignError(f"{name}: arm domain differs")
        numel = math.prod(entry.source_weight_shape)
        for arm_name, arm in arms.items():
            if not isinstance(arm, Mapping):
                raise CampaignError(f"{name}.{arm_name}: arm must be an object")
            expected_arm_keys = (
                _CB_LEARNED_ARM_KEYS
                if arm_name.startswith("fp8_cb_learned@")
                else _CB_ARM_KEYS
                if arm_name.startswith("fp8_cb_fixed@")
                else _TCQ_ARM_KEYS
            )
            _exact_keys(
                arm, expected_arm_keys, where=f"{name}.{arm_name}"
            )
            if arm_name.startswith("fp8_cb_"):
                rung = int(arm_name.rsplit("@", 1)[1])
                book_kind = (
                    "per_tensor_weighted_lloyd"
                    if arm_name.startswith("fp8_cb_learned@")
                    else "fixed_lattice"
                )
                if (
                    arm.get("family") != "FP8_CB_K"
                    or arm.get("rung") != rung
                    or arm.get("encode_tier") != ENCODE_TIER
                    or arm.get("book_kind") != book_kind
                ):
                    raise CampaignError(f"{name}.{arm_name}: CB identity differs")
            else:
                stem, rate_text = arm_name.rsplit("@", 1)
                _family, bracket, selector = stem.split(".")
                if (
                    arm.get("family") != "TCQ_E4M3_R256"
                    or arm.get("rate") != float(rate_text)
                    or arm.get("trellis_scale_bracket") != bracket
                    or arm.get("alphabet_selector") != selector
                ):
                    raise CampaignError(f"{name}.{arm_name}: TCQ identity differs")
                if not _SHA256_RE.fullmatch(str(arm.get("e4m3_plane_sha256"))):
                    raise CampaignError(f"{name}.{arm_name}: plane hash invalid")
            error = _finite(
                arm.get("weighted_sse"),
                where=f"{name}.{arm_name}.weighted_sse",
                nonnegative=True,
            )
            _finite(
                arm.get("encode_seconds_observation_not_perf_claim"),
                where=f"{name}.{arm_name}.encode_seconds",
                nonnegative=True,
            )
            nsse = _finite(
                arm.get("weighted_nsse"), where=f"{name}.{arm_name}.weighted_nsse"
            )
            snr = _finite(
                arm.get("weighted_snr_db"), where=f"{name}.{arm_name}.weighted_snr_db"
            )
            if not math.isclose(nsse, error / energy, rel_tol=1e-12, abs_tol=1e-15):
                raise CampaignError(f"{name}.{arm_name}: NSSE identity differs")
            expected_snr = -10.0 * math.log10(max(nsse, 1e-300))
            if not math.isclose(snr, expected_snr, rel_tol=1e-12, abs_tol=1e-12):
                raise CampaignError(f"{name}.{arm_name}: SNR identity differs")
            if not _SHA256_RE.fullmatch(str(arm.get("reconstruction_sha256"))):
                raise CampaignError(f"{name}.{arm_name}: reconstruction hash invalid")
            _validate_footprint(arm, numel=numel, where=f"{name}.{arm_name}")
    if require_complete and partial is not False:
        raise CampaignError("completed result required")
    if partial is False:
        if names != expected_names:
            raise CampaignError("final result does not cover the full corpus")
        if report.get("status") != STATUS or report.get("claim_boundary") != CLAIM_BOUNDARY:
            raise CampaignError("final no-serving claim boundary differs")
        expected_summaries = population_summaries(per_tensor)
        if report.get("population_summaries") != expected_summaries:
            raise CampaignError("population summaries are not exactly derived")
        if set(expected_summaries) != {"dense", "routed"}:
            raise CampaignError("final summaries must contain only dense and routed")


def _resume(path: Path, *, settings, entries) -> dict[str, object]:
    sealed = _strict_report(path)
    validate_report(sealed, settings=settings, entries=entries)
    return {key: value for key, value in sealed.items() if key != "checkpoint_sha256"}


def _verify_final_bindings(*, args, settings, ladder) -> None:
    if settings.get("environment") != _execution_environment(ladder):
        raise CampaignError("numeric execution environment drifted during run")
    if settings.get("active_source_identity") != _active_source_identity():
        raise CampaignError("active source identity drifted during run")
    if settings.get("locked_sources") != _locked_sources(args.locked_ladder):
        raise CampaignError("frozen ladder identity drifted during run")
    if settings.get("frozen_codec_closure") != BASE._frozen_codec_closure(ladder):
        raise CampaignError("frozen codec closure drifted during run")
    fresh, binding = load_active_glm_corpus_bound(REPO_ROOT, args.manifest)
    if (
        binding["sha256"] != settings.get("corpus_manifest_sha256")
        or file_sha256(fresh.artifact_path) != settings.get("corpus_file_sha256")
        or fresh.manifest["importance_identity"]["value_sha256"]
        != settings.get("importance_value_sha256")
        or fresh.manifest["prismaquant_commit"]
        != settings.get("corpus_prismaquant_commit")
    ):
        raise CampaignError("bound GLM corpus drifted during run")


def _run(args, corpus, settings, ladder) -> int:
    import torch

    if not torch.cuda.is_available():
        raise CampaignError("CUDA is required; this GPU campaign has no CPU fallback")
    if torch.cuda.device_count() != 1:
        raise CampaignError("exactly one visible CUDA device is required")
    if args.out.exists() or args.out.is_symlink():
        raise CampaignError("final output already exists (immutable no-clobber)")
    partial = args.out.with_name(args.out.name + ".partial")
    if partial.is_symlink():
        raise CampaignError("partial checkpoint must not be a symlink")
    if partial.exists():
        report = _resume(partial, settings=settings, entries=corpus.entries)
    else:
        report = {
            "schema": SCHEMA,
            "settings": settings,
            "started_at_unix_s": time.time(),
            "per_tensor": {},
            "partial": True,
            "tensors_done": 0,
        }
    saved = dict(report["per_tensor"])
    report["per_tensor"] = {}
    device = torch.device("cuda:0")
    for index, entry in enumerate(corpus.entries, 1):
        cell = _measure_cell(
            entry=entry, corpus=corpus, ladder=ladder, device=device
        )
        if entry.name in saved:
            require_replay_match(entry.name, saved[entry.name], cell)
            suffix = " REPLAY VERIFIED"
        else:
            suffix = ""
        report["per_tensor"][entry.name] = cell
        report["tensors_done"] = index
        sealed = _sealed(report)
        validate_report(sealed, settings=settings, entries=corpus.entries)
        atomic_checkpoint_json(partial, sealed)
        print(
            f"[{index}/{len(corpus.entries)}] {entry.population} {entry.name}{suffix}",
            flush=True,
        )
        torch.cuda.empty_cache()
    report.update({
        "partial": False,
        "completed_at_unix_s": time.time(),
        "population_summaries": population_summaries(report["per_tensor"]),
        "status": STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    })
    sealed = _sealed(report)
    validate_report(
        sealed, settings=settings, entries=corpus.entries, require_complete=True
    )
    atomic_checkpoint_json(partial, sealed)
    _verify_final_bindings(args=args, settings=settings, ladder=ladder)
    publish_file_no_replace(partial, args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locked-ladder", type=Path, default=DEFAULT_LOCKED_LADDER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    corpus, manifest_binding = load_active_glm_corpus_bound(REPO_ROOT, args.manifest)
    locked = _locked_sources(args.locked_ladder)
    if locked != _locked_sources(args.locked_ladder):
        raise CampaignError("locked source identity changed during preflight")
    if args.preflight_only:
        preflight = {
            "schema": SCHEMA,
            "status": "validated_no_gpu_no_write",
            "publication_capable": False,
            "publication_receipt": None,
            "corpus_manifest_sha256": manifest_binding["sha256"],
            "corpus_file_sha256": corpus.manifest["file_sha256"],
            "population_counts": {
                name: len(entries) for name, entries in corpus.populations.items()
            },
            "rungs": list(RUNGS),
            "rates": list(RATES),
            "trellis_scale_brackets": list(TRELLIS_BRACKETS),
            "alphabet_selectors": list(ALPHABET_SELECTORS),
            "book_price_brackets": list(BOOK_PRICE_BRACKETS),
            "encode_tier": ENCODE_TIER,
            "locked_sources": locked,
            "active_source_identity": _active_source_identity(),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    ladder = BASE._load_ladder(args.locked_ladder)
    execution = _execution_environment(ladder)
    settings = _settings(
        args=args,
        corpus=corpus,
        manifest_binding=manifest_binding,
        ladder=ladder,
        execution=execution,
    )
    try:
        with exclusive_publication_claim(args.out, identity=settings):
            return _run(args, corpus, settings, ladder)
    except PublicationError as exc:
        raise CampaignError(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
