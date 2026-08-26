"""Numerical ship gate for a materialized PrismaSnap BF16 checkpoint.

Materialization proves the exact transform program, shard census, and fp64
algebra.  It deliberately does not call that output numerically VERIFIED.
This module consumes the standard ``measure_vllm_full_kl.py`` student result
against the original BF16 teacher and performs the only MATERIALIZED ->
VERIFIED transition admitted by the native exporter.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import torch

from .cost_stage_checkpoint import canonical_json_sha256
from .cost_streaming import (
    canonical_streamed_model_semantic_config,
    portable_streamed_model_content_identity,
    validate_streamed_model_identity,
)
from .prismasnap import (
    PRISMASNAP_ALGORITHM,
    PRISMASNAP_BF16_REALIZATION_POLICY,
    PRISMASNAP_BF16_REALIZED_ALGORITHM,
    PrismaSnapSearchConfig,
)
from .prismasnap_checkpoint import (
    BF16_REALIZATION_SCHEMA,
    PLAN_SET_SCHEMA,
    _BF16_DERIVATION_KEYS,
    _BF16_DERIVATION_PARENT_KEYS,
    _BF16_DERIVATION_SOURCE_KEYS,
    _BF16_REALIZED_NORM_KEYS,
    _BF16_REALIZED_UPDOWN_KEYS,
    _derivation_digest,
)


PROVENANCE_SCHEMA = "prismaquant.prismasnap.provenance.v1"
PROVENANCE_SCHEMA_V2 = "prismaquant.prismasnap.provenance.v2"
PROVENANCE_JSON = "prismasnap_provenance.json"
FOLD_FIDELITY_SCHEMA = "prismaquant.prismasnap.fold_fidelity.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BASE_PROVENANCE_KEYS = frozenset(
    {
        "schema",
        "state",
        "algorithm",
        "purely_additive_source_preparation",
        "serve_time_changes",
        "source_portable_content_sha256",
        "source_local_content_sha256",
        "source_model",
        "probe_sha256",
        "calibration",
        "plan_sha256",
        "scales_sha256",
        "producer",
        "search",
        "coverage",
        "fp64_invariance",
        "seam_summary",
        "output",
        "provenance_sha256",
    }
)
_FOLD_KEYS = frozenset(
    {
        "schema",
        "passed",
        "metric",
        "kl_mean",
        "threshold",
        "score_positions",
        "prompt_top_k",
        "n_samples",
        "seqlen",
        "vocab_size",
        "student_result_sha256",
        "teacher_meta_sha256",
        "teacher_payload_sha256",
        "source_identity_sha256",
        "source_portable_content_sha256",
        "source_local_content_sha256",
        "calibration_ids_sha256",
        "calibration_starts",
        "calibration_corpus_sha256",
        "checkpoint_shard_content_sha256",
        "checkpoint_weight_map_sha256",
        "checkpoint_index_sha256",
        "materialized_provenance_sha256",
        "serve_fingerprint",
        "teacher_serve_fingerprint",
    }
)


def _require_exact_mapping(
    value: object, keys: set[str] | frozenset[str], *, where: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        observed = set(value) if isinstance(value, Mapping) else set()
        raise RuntimeError(
            f"{where} fields differ; missing={sorted(set(keys) - observed)} "
            f"extra={sorted(observed - set(keys))}"
        )
    return value


def _require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeError(f"{where} is not a full lowercase SHA-256")
    return value


def _require_positive_int(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{where} must be a positive integer")
    return value


def _is_number(value: object) -> bool:
    return type(value) in {int, float}


def _validate_bf16_derivation(value: object, *, where: str) -> None:
    derivation = _require_exact_mapping(value, _BF16_DERIVATION_KEYS, where=where)
    if (
        derivation.get("schema") != BF16_REALIZATION_SCHEMA
        or derivation.get("policy") != PRISMASNAP_BF16_REALIZATION_POLICY
        or derivation.get("nominal_search_stats_semantics")
        != "copied_parent_v1_nominal_search_not_executed"
        or derivation.get("realized_execution_metrics_semantics")
        != "exact_static6_nvfp4_on_executed_bf16_consumer_weights"
        or derivation.get("derivation_sha256") != _derivation_digest(derivation)
    ):
        raise RuntimeError(f"{where} contract failed")
    parent = _require_exact_mapping(
        derivation.get("parent"),
        _BF16_DERIVATION_PARENT_KEYS,
        where=f"{where}.parent",
    )
    if (
        parent.get("schema") != PLAN_SET_SCHEMA
        or parent.get("algorithm") != PRISMASNAP_ALGORITHM
        or not isinstance(parent.get("producer"), Mapping)
    ):
        raise RuntimeError(f"{where} parent is not merged v1")
    _require_sha256(parent.get("plan_sha256"), where=f"{where}.parent.plan")
    _require_sha256(parent.get("scales_sha256"), where=f"{where}.parent.scales")
    source = _require_exact_mapping(
        derivation.get("source"),
        _BF16_DERIVATION_SOURCE_KEYS,
        where=f"{where}.source",
    )
    _require_sha256(
        source.get("local_content_sha256"), where=f"{where}.source.local"
    )
    _require_sha256(
        source.get("portable_content_sha256"), where=f"{where}.source.portable"
    )
    seams = derivation.get("seams")
    if not isinstance(seams, list) or not seams:
        raise RuntimeError(f"{where} has no seam execution records")
    seen: set[tuple[int, str]] = set()
    for ordinal, raw in enumerate(seams):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"{where}.seams[{ordinal}] is malformed")
        kind = raw.get("kind")
        keys = (
            _BF16_REALIZED_NORM_KEYS
            if kind in {"input_norm", "post_attention_norm"}
            else _BF16_REALIZED_UPDOWN_KEYS
        )
        row = _require_exact_mapping(raw, keys, where=f"{where}.seams[{ordinal}]")
        layer = row.get("layer")
        if type(layer) is not int or layer < 0 or kind not in {
            "input_norm",
            "post_attention_norm",
            "up_down",
        }:
            raise RuntimeError(f"{where}.seams[{ordinal}] key is malformed")
        key = (layer, str(kind))
        if key in seen:
            raise RuntimeError(f"{where} repeats a seam")
        seen.add(key)
        _require_sha256(row.get("graph_sha256"), where=f"{where} graph")
        for name in (
            "nominal_vector_sha256",
            "executed_vector_sha256",
        ):
            _require_sha256(row.get(name), where=f"{where}.{name}")
        channels = row.get("channels")
        executed_channels = row.get("executed_realized_channels")
        executed_groups = row.get("executed_groups_moved")
        group_size = PrismaSnapSearchConfig().group_size
        if (
            type(channels) is not int
            or channels <= 0
            or channels % group_size != 0
            or type(executed_channels) is not int
            or executed_channels < 0
            or executed_channels > channels
            or type(executed_groups) is not int
            or executed_groups < 0
            or executed_groups > channels // group_size
        ):
            raise RuntimeError(f"{where} execution census is malformed")
        if kind == "up_down":
            if (
                executed_channels != 0
                or executed_groups != 0
                or row.get("policy_reason")
                != "disabled_for_bf16_fold_fidelity_v2"
            ):
                raise RuntimeError(f"{where} up/down was not forced to identity")
            continue
        _require_sha256(
            row.get("projected_norm_sha256"), where=f"{where} projected norm"
        )
        count_names = (
            "candidate_realized_channels",
            "source_zero_channels",
            "projected_zero_channels",
            "sign_mismatch_channels",
            "nonfinite_channels",
            "reapplication_mismatch_channels",
        )
        if any(
            type(row.get(name)) is not int
            or int(row[name]) < 0
            or int(row[name]) > channels
            for name in count_names
        ):
            raise RuntimeError(f"{where} projection reason counts are malformed")
        baseline = row.get("objective_baseline")
        realized = row.get("objective_realized")
        improvement = row.get("objective_improvement_fraction")
        accepted = row.get("objective_accepted")
        if (
            not _is_number(baseline)
            or not math.isfinite(float(baseline))
            or float(baseline) < 0.0
            or not _is_number(realized)
            or not math.isfinite(float(realized))
            or float(realized) < 0.0
            or float(realized) > float(baseline)
            or not _is_number(improvement)
            or not math.isfinite(float(improvement))
            or type(accepted) is not bool
            or (accepted and not float(realized) < float(baseline))
            or (not accepted and float(realized) != float(baseline))
            or row.get("objective_fallback_reason")
            != (None if accepted else "no_strict_realized_bf16_improvement")
            or row.get("projected_norm_materialization") != "replace_bf16"
            or row.get("consumer_inverse_materialization")
            != "divide_then_round_bf16"
        ):
            raise RuntimeError(f"{where} realized objective is malformed")
        expected_improvement = (
            0.0
            if float(baseline) == 0.0
            else (float(baseline) - float(realized)) / float(baseline)
        )
        if not math.isclose(
            float(improvement), expected_improvement, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise RuntimeError(f"{where} realized objective is incoherent")
        if accepted and (
            executed_channels != row.get("candidate_realized_channels")
            or executed_groups <= 0
        ):
            raise RuntimeError(f"{where} accepted execution census differs")
        if not accepted and (executed_channels != 0 or executed_groups != 0):
            raise RuntimeError(f"{where} fallback did not execute identity")


def _read_regular_bytes(path: Path, *, where: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{where} is not a regular file: {path}")
    return path.read_bytes()


def _json_from_bytes(data: bytes, *, where: str) -> dict[str, Any]:
    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate member {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=exact_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {value}")
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"{where} is corrupt") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{where} must be a JSON object")
    return value


def _load_json(path: Path, *, where: str) -> dict[str, Any]:
    return _json_from_bytes(
        _read_regular_bytes(path, where=where), where=f"{where} {path}"
    )


def _sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"PrismaSnap evidence is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _provenance_digest(payload: Mapping[str, object]) -> str:
    return canonical_json_sha256(
        {str(key): value for key, value in payload.items() if key != "provenance_sha256"},
        where="PrismaSnap provenance",
    )


def validate_prismasnap_provenance_payload(
    payload: Mapping[str, object],
    *,
    require_verified: bool,
    where: str,
) -> None:
    state = payload.get("state")
    expected_keys = set(_BASE_PROVENANCE_KEYS)
    schema = payload.get("schema")
    if schema == PROVENANCE_SCHEMA_V2:
        expected_keys.add("derivation")
    if "collation" in payload:
        expected_keys.add("collation")
    if state == "VERIFIED":
        expected_keys.add("fold_fidelity")
    _require_exact_mapping(payload, expected_keys, where=where)
    if schema not in {PROVENANCE_SCHEMA, PROVENANCE_SCHEMA_V2}:
        raise RuntimeError(f"{where} has an unsupported schema")
    allowed = {"VERIFIED"} if require_verified else {"MATERIALIZED", "VERIFIED"}
    if state not in allowed:
        raise RuntimeError(f"{where} is not in an admitted state: {state!r}")
    claimed = payload.get("provenance_sha256")
    if not isinstance(claimed, str) or claimed != _provenance_digest(payload):
        raise RuntimeError(f"{where} self digest mismatch")
    if (
        payload.get("purely_additive_source_preparation") is not True
        or payload.get("serve_time_changes") is not False
    ):
        raise RuntimeError(f"{where} additive-source contract failed")
    expected_algorithm = (
        PRISMASNAP_BF16_REALIZED_ALGORITHM
        if schema == PROVENANCE_SCHEMA_V2
        else PRISMASNAP_ALGORITHM
    )
    if payload.get("algorithm") != expected_algorithm:
        raise RuntimeError(f"{where} names an unrecognized algorithm")
    for key in (
        "source_portable_content_sha256",
        "source_local_content_sha256",
        "probe_sha256",
        "plan_sha256",
        "scales_sha256",
    ):
        _require_sha256(payload.get(key), where=f"{where}.{key}")
    if schema == PROVENANCE_SCHEMA_V2:
        _validate_bf16_derivation(
            payload.get("derivation"), where=f"{where}.derivation"
        )
        derivation = payload["derivation"]
        assert isinstance(derivation, Mapping)
        derivation_source = derivation["source"]
        assert isinstance(derivation_source, Mapping)
        if (
            derivation_source.get("local_content_sha256")
            != payload.get("source_local_content_sha256")
            or derivation_source.get("portable_content_sha256")
            != payload.get("source_portable_content_sha256")
        ):
            raise RuntimeError(f"{where} derivation source binding differs")
    if not isinstance(payload.get("source_model"), str) or not payload["source_model"]:
        raise RuntimeError(f"{where} has no source model path")

    producer = _require_exact_mapping(
        payload.get("producer"),
        {
            "git_commit",
            "source_sha256",
            "source_files",
            "container_rootfs_sha256",
            "container_attested",
        },
        where=f"{where}.producer",
    )
    if (
        not isinstance(producer.get("git_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(producer["git_commit"])) is None
    ):
        raise RuntimeError(f"{where} producer commit is malformed")
    source_files = producer.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise RuntimeError(f"{where} producer source-file closure is empty")
    normalized_files: dict[str, str] = {}
    for name, digest in source_files.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise RuntimeError(f"{where} producer source path is unsafe")
        normalized_files[name] = _require_sha256(
            digest, where=f"{where}.producer.source_files[{name!r}]"
        )
    if _require_sha256(
        producer.get("source_sha256"), where=f"{where}.producer.source_sha256"
    ) != canonical_json_sha256(
        normalized_files, where="PrismaSnap producer source files"
    ):
        raise RuntimeError(f"{where} producer source closure digest differs")
    container_attested = producer.get("container_attested")
    container_rootfs = producer.get("container_rootfs_sha256")
    if container_attested is not True:
        raise RuntimeError(f"{where} producer container is not attested")
    _require_sha256(container_rootfs, where=f"{where} producer container rootfs")

    expected_search = PrismaSnapSearchConfig().as_dict()
    if payload.get("search") != expected_search:
        raise RuntimeError(
            f"{where} does not carry the measured-fast stage,polish treatment"
        )

    calibration = _require_exact_mapping(
        payload.get("calibration"),
        {"calib_hash", "dataset", "nsamples", "seqlen", "calibration_modality"},
        where=f"{where}.calibration",
    )
    calib_hash = calibration.get("calib_hash")
    dataset = calibration.get("dataset")
    modality = calibration.get("calibration_modality")
    if (
        not isinstance(calib_hash, str)
        or not calib_hash
        or not isinstance(dataset, str)
        or not dataset
        or _require_positive_int(
            calibration.get("nsamples"), where=f"{where}.calibration.nsamples"
        )
        < 2
        or _require_positive_int(
            calibration.get("seqlen"), where=f"{where}.calibration.seqlen"
        )
        < 512
        or (modality is not None and modality not in {"text", "text_only", "text-only"})
    ):
        raise RuntimeError(f"{where} calibration contract is malformed")

    coverage = _require_exact_mapping(
        payload.get("coverage"),
        {
            "body_layers",
            "excluded_prefixes",
            "seams",
            "transformed_tensors",
            "materialized_changed_tensors",
        },
        where=f"{where}.coverage",
    )
    body_layers = coverage.get("body_layers")
    if (
        not isinstance(body_layers, list)
        or not body_layers
        or any(type(layer) is not int or layer < 0 for layer in body_layers)
        or body_layers != list(range(len(body_layers)))
        or coverage.get("excluded_prefixes") != ["model.visual.", "mtp."]
        or coverage.get("seams") != 3 * len(body_layers)
    ):
        raise RuntimeError(f"{where} dense body coverage is not exact")
    raw_transformed = coverage.get("transformed_tensors")
    if schema == PROVENANCE_SCHEMA_V2:
        if type(raw_transformed) is not int or raw_transformed < 0:
            raise RuntimeError(
                f"{where}.coverage.transformed_tensors must be nonnegative"
            )
        transformed = raw_transformed
    else:
        transformed = _require_positive_int(
            raw_transformed,
            where=f"{where}.coverage.transformed_tensors",
        )
    if coverage.get("materialized_changed_tensors") != transformed:
        raise RuntimeError(f"{where} transformed-tensor census differs")

    verification = _require_exact_mapping(
        payload.get("fp64_invariance"),
        {
            "fp64_invariance_max_abs",
            "threshold",
            "domain",
            "required_bf16_fold_kl_max",
        },
        where=f"{where}.fp64_invariance",
    )
    error = verification.get("fp64_invariance_max_abs")
    if (
        not _is_number(error)
        or not math.isfinite(float(error))
        or float(error) < 0.0
        or float(error) > 1e-10
        or verification.get("threshold") != 1e-10
        or verification.get("domain") != "pre_cast_fp64_algebra"
        or verification.get("required_bf16_fold_kl_max") != 5e-4
    ):
        raise RuntimeError(f"{where} fp64 invariance contract failed")

    summaries = payload.get("seam_summary")
    if not isinstance(summaries, list) or len(summaries) != 3 * len(body_layers):
        raise RuntimeError(f"{where} seam-summary census differs")
    kinds_by_layer: dict[int, set[str]] = {layer: set() for layer in body_layers}
    graphs_by_layer: dict[int, set[str]] = {layer: set() for layer in body_layers}
    for ordinal, raw in enumerate(summaries):
        row = _require_exact_mapping(
            raw,
            {
                "layer",
                "kind",
                "graph_sha256",
                "groups",
                "groups_moved",
                "improvement_fraction",
            },
            where=f"{where}.seam_summary[{ordinal}]",
        )
        layer = row.get("layer")
        kind = row.get("kind")
        groups = row.get("groups")
        moved = row.get("groups_moved")
        improvement = row.get("improvement_fraction")
        if (
            type(layer) is not int
            or layer not in kinds_by_layer
            or kind not in {"input_norm", "post_attention_norm", "up_down"}
            or type(groups) is not int
            or groups <= 0
            or type(moved) is not int
            or moved < 0
            or moved > groups
            or not _is_number(improvement)
            or not math.isfinite(float(improvement))
            or float(improvement) < 0.0
            or float(improvement) > 1.0
        ):
            raise RuntimeError(f"{where} seam summary row is malformed")
        kinds_by_layer[layer].add(str(kind))
        graphs_by_layer[layer].add(
            _require_sha256(row.get("graph_sha256"), where=f"{where} seam graph")
        )
    if any(
        kinds != {"input_norm", "post_attention_norm", "up_down"}
        or len(graphs_by_layer[layer]) != 1
        for layer, kinds in kinds_by_layer.items()
    ):
        raise RuntimeError(f"{where} seam graph coverage is not one exact trio per layer")
    if schema == PROVENANCE_SCHEMA_V2:
        derivation = payload["derivation"]
        assert isinstance(derivation, Mapping)
        realized_seams = derivation["seams"]
        assert isinstance(realized_seams, list)
        realized_by_key = {
            (int(row["layer"]), str(row["kind"]), str(row["graph_sha256"])): row
            for row in realized_seams
            if isinstance(row, Mapping)
        }
        summary_by_key = {
            (int(row["layer"]), str(row["kind"]), str(row["graph_sha256"])): row
            for row in summaries
            if isinstance(row, Mapping)
        }
        if (
            set(realized_by_key) != set(summary_by_key)
            or len(realized_by_key) != len(realized_seams)
            or len(summary_by_key) != len(summaries)
        ):
            raise RuntimeError(f"{where} realized/nominal seam bindings differ")
        group_size = PrismaSnapSearchConfig().group_size
        for key, realized_row in realized_by_key.items():
            summary_row = summary_by_key[key]
            kind = str(realized_row["kind"])
            expected_improvement = (
                0.0
                if kind == "up_down"
                else float(realized_row["objective_improvement_fraction"])
            )
            if (
                summary_row["groups"]
                != int(realized_row["channels"]) // group_size
                or summary_row["groups_moved"]
                != realized_row["executed_groups_moved"]
                or float(summary_row["improvement_fraction"])
                != expected_improvement
            ):
                raise RuntimeError(
                    f"{where} realized seam summary differs from execution"
                )

    output = _require_exact_mapping(
        payload.get("output"),
        {
            "tensors",
            "shards",
            "checkpoint_weight_map_sha256",
            "index_sha256",
            "shard_content_sha256",
        },
        where=f"{where}.output",
    )
    _require_positive_int(output.get("tensors"), where=f"{where}.output.tensors")
    _require_positive_int(output.get("shards"), where=f"{where}.output.shards")
    for key in (
        "checkpoint_weight_map_sha256",
        "index_sha256",
        "shard_content_sha256",
    ):
        _require_sha256(output.get(key), where=f"{where}.output.{key}")

    if "collation" in payload:
        _validate_collation(payload["collation"], output=output, where=f"{where}.collation")

    if state == "VERIFIED":
        fold = payload.get("fold_fidelity")
        fold = _require_exact_mapping(fold, _FOLD_KEYS, where=f"{where}.fold_fidelity")
        value = fold.get("kl_mean")
        limit = verification.get("required_bf16_fold_kl_max")
        if (
            fold.get("schema") != FOLD_FIDELITY_SCHEMA
            or fold.get("passed") is not True
            or fold.get("metric")
            != "forward_kl_original_bf16_to_snapped_bf16"
            or fold.get("score_positions") != "all"
            or not _is_number(value)
            or not _is_number(limit)
            or not math.isfinite(float(value))
            or not math.isfinite(float(limit))
            or float(value) < 0.0
            or float(value) > float(limit)
            or not _is_number(fold.get("threshold"))
            or float(fold["threshold"]) != float(limit)
        ):
            raise RuntimeError(f"{where} fold-fidelity gate failed")
        for key in (
            "student_result_sha256",
            "teacher_meta_sha256",
            "teacher_payload_sha256",
            "source_identity_sha256",
            "source_portable_content_sha256",
            "source_local_content_sha256",
            "checkpoint_shard_content_sha256",
            "checkpoint_weight_map_sha256",
            "checkpoint_index_sha256",
            "materialized_provenance_sha256",
            "calibration_ids_sha256",
            "calibration_corpus_sha256",
            "serve_fingerprint",
            "teacher_serve_fingerprint",
        ):
            if not isinstance(fold.get(key), str) or _SHA256.fullmatch(str(fold[key])) is None:
                raise RuntimeError(f"{where} has malformed fold evidence {key}")
        starts = fold.get("calibration_starts")
        n_samples = fold.get("n_samples")
        if (
            type(n_samples) is not int
            or n_samples < 2
            or type(fold.get("prompt_top_k")) is not int
            or int(fold["prompt_top_k"]) <= 0
            or type(fold.get("seqlen")) is not int
            or int(fold["seqlen"]) < 512
            or type(fold.get("vocab_size")) is not int
            or int(fold["vocab_size"]) <= 0
            or not isinstance(starts, list)
            or len(starts) != n_samples
            or any(type(value) is not int or value < 0 for value in starts)
        ):
            raise RuntimeError(f"{where} has malformed calibration windows")
        if (
            fold.get("source_portable_content_sha256")
            != payload["source_portable_content_sha256"]
            or fold.get("source_local_content_sha256")
            != payload["source_local_content_sha256"]
            or fold.get("checkpoint_shard_content_sha256")
            != output["shard_content_sha256"]
            or fold.get("checkpoint_weight_map_sha256")
            != output["checkpoint_weight_map_sha256"]
            or fold.get("checkpoint_index_sha256") != output["index_sha256"]
        ):
            raise RuntimeError(f"{where} fold evidence belongs to different content")


def _validate_collation(
    value: object, *, output: Mapping[str, object], where: str
) -> None:
    collation = _require_exact_mapping(
        value,
        {
            "parts",
            "ordered_part_bindings",
            "ordered_part_bindings_sha256",
            "source_metadata",
            "source_metadata_sha256",
            "shard_transfer_strategy",
            "exact_disjoint_shard_union",
        },
        where=where,
    )
    parts = collation.get("parts")
    bindings = collation.get("ordered_part_bindings")
    if (
        not isinstance(parts, list)
        or not parts
        or any(_SHA256.fullmatch(str(item)) is None for item in parts)
        or len(set(parts)) != len(parts)
        or not isinstance(bindings, list)
        or len(bindings) != len(parts)
        or collation.get("shard_transfer_strategy")
        not in {"hardlink_required", "durable_copy"}
        or collation.get("exact_disjoint_shard_union") is not True
    ):
        raise RuntimeError(f"{where} part census is malformed")
    seen_shards: set[str] = set()
    for ordinal, raw in enumerate(bindings):
        binding = _require_exact_mapping(
            raw,
            {"ordinal", "part_sha256", "manifest_file_sha256", "shards"},
            where=f"{where}.ordered_part_bindings[{ordinal}]",
        )
        if binding.get("ordinal") != ordinal or binding.get("part_sha256") != parts[ordinal]:
            raise RuntimeError(f"{where} ordered part binding differs")
        _require_sha256(binding.get("manifest_file_sha256"), where=f"{where} manifest")
        shards = binding.get("shards")
        if not isinstance(shards, list) or not shards:
            raise RuntimeError(f"{where} part has no shards")
        for row_ordinal, raw_row in enumerate(shards):
            row = _require_exact_mapping(
                raw_row,
                {
                    "source_name",
                    "source_bytes",
                    "source_sha256",
                    "output_bytes",
                    "output_sha256",
                    "tensor_count",
                    "changed_tensors",
                    "receipt_sha256",
                },
                where=f"{where}.part[{ordinal}].shards[{row_ordinal}]",
            )
            name = row.get("source_name")
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.endswith(".safetensors")
                or name in seen_shards
            ):
                raise RuntimeError(f"{where} shard ownership is not disjoint")
            seen_shards.add(name)
            for key in ("source_bytes", "output_bytes", "tensor_count"):
                _require_positive_int(row.get(key), where=f"{where}.{key}")
            if type(row.get("changed_tensors")) is not int or int(row["changed_tensors"]) < 0:
                raise RuntimeError(f"{where} changed-tensor census is malformed")
            for key in ("source_sha256", "output_sha256", "receipt_sha256"):
                _require_sha256(row.get(key), where=f"{where}.{key}")
    if len(seen_shards) != output.get("shards"):
        raise RuntimeError(f"{where} shard union differs from checkpoint output")
    if _require_sha256(
        collation.get("ordered_part_bindings_sha256"), where=f"{where} part digest"
    ) != canonical_json_sha256(bindings, where="PrismaSnap ordered part bindings"):
        raise RuntimeError(f"{where} ordered part binding digest differs")

    metadata = collation.get("source_metadata")
    if not isinstance(metadata, list) or not metadata:
        raise RuntimeError(f"{where} source metadata closure is empty")
    names: set[str] = set()
    for ordinal, raw in enumerate(metadata):
        row = _require_exact_mapping(
            raw, {"name", "bytes", "sha256"}, where=f"{where}.source_metadata[{ordinal}]"
        )
        name = row.get("name")
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise RuntimeError(f"{where} source metadata names are malformed")
        names.add(name)
        _require_positive_int(row.get("bytes"), where=f"{where} metadata bytes")
        _require_sha256(row.get("sha256"), where=f"{where} metadata digest")
    if _require_sha256(
        collation.get("source_metadata_sha256"), where=f"{where} metadata closure"
    ) != canonical_json_sha256(metadata, where="PrismaSnap source metadata bindings"):
        raise RuntimeError(f"{where} source metadata digest differs")


def _checkpoint_content_identity(root: Path) -> dict[str, object]:
    index = _load_json(
        root / "model.safetensors.index.json", where="PrismaSnap checkpoint index"
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("PrismaSnap checkpoint index has no weight_map")
    if not all(
        isinstance(key, str)
        and key
        and isinstance(name, str)
        and Path(name).name == name
        and name.endswith(".safetensors")
        for key, name in weight_map.items()
    ):
        raise RuntimeError("PrismaSnap checkpoint index shard names are malformed")
    shards = sorted(set(weight_map.values()))
    entries = [path for path in root.iterdir() if path.suffix == ".safetensors"]
    unsafe = [path.name for path in entries if path.is_symlink() or not path.is_file()]
    if unsafe or {path.name for path in entries} != set(shards):
        raise RuntimeError("PrismaSnap checkpoint shard-file census changed")
    rows = [
        {
            "name": name,
            "size": (root / name).stat().st_size,
            "sha256": _sha256_file(root / name),
        }
        for name in shards
    ]
    return {
        "tensors": len(weight_map),
        "shards": len(shards),
        "shard_content_sha256": canonical_json_sha256(
            rows, where="PrismaSnap output shard identity"
        ),
        "checkpoint_weight_map_sha256": canonical_json_sha256(
            dict(sorted(weight_map.items())),
            where="PrismaSnap output checkpoint weight map",
        ),
        "index_sha256": _sha256_file(root / "model.safetensors.index.json"),
    }


def validate_prismasnap_checkpoint(
    checkpoint_dir: str | Path,
    *,
    require_verified: bool = True,
) -> dict[str, object]:
    """Replay provenance plus current index/shard bytes at an admission point."""
    root = Path(checkpoint_dir)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"PrismaSnap checkpoint is not a real directory: {root}")
    root = root.resolve(strict=True)
    provenance = _load_json(root / PROVENANCE_JSON, where="PrismaSnap provenance")
    validate_prismasnap_provenance_payload(
        provenance,
        require_verified=require_verified,
        where=f"PrismaSnap checkpoint {root}",
    )
    observed = _checkpoint_content_identity(root)
    output = provenance.get("output")
    if not isinstance(output, Mapping) or any(
        output.get(key) != value for key, value in observed.items()
    ):
        raise RuntimeError("PrismaSnap checkpoint content differs from provenance")
    if require_verified:
        fold = provenance.get("fold_fidelity")
        if not isinstance(fold, Mapping) or (
            fold.get("checkpoint_shard_content_sha256")
            != observed["shard_content_sha256"]
        ):
            raise RuntimeError("PrismaSnap checkpoint content differs from fold evidence")
    return dict(provenance)


def _same_resolved_path(left: object, right: Path, *, where: str) -> None:
    if not isinstance(left, str):
        raise RuntimeError(f"{where} path is missing")
    try:
        resolved = Path(left).resolve(strict=True)
    except Exception as exc:
        raise RuntimeError(f"{where} path is not locally resolvable: {left!r}") from exc
    if resolved != right:
        raise RuntimeError(f"{where} path belongs to another model")


def _validated_source_identity_binding(
    source_root: Path, identity_path: Path
) -> tuple[dict[str, object], dict[str, object], str]:
    """Validate the original source bytes without trusting a path-only stamp."""
    identity_bytes = _read_regular_bytes(
        identity_path, where="PrismaSnap original-source identity"
    )
    raw = _json_from_bytes(
        identity_bytes, where="PrismaSnap original-source identity"
    )
    identity = validate_streamed_model_identity(
        raw, where="PrismaSnap original-source identity"
    )
    _same_resolved_path(
        identity.get("source"), source_root, where="source identity model"
    )
    index = _load_json(
        source_root / "model.safetensors.index.json",
        where="PrismaSnap original-source checkpoint index",
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("PrismaSnap original source has no checkpoint weight map")
    if identity.get("checkpoint_weight_map") != dict(sorted(weight_map.items())):
        raise RuntimeError("PrismaSnap original source index differs from its identity")

    config_path = source_root / "config.json"
    before_config = (config_path.stat().st_size, _sha256_file(config_path))
    try:
        from transformers import AutoConfig

        live_config = canonical_streamed_model_semantic_config(
            AutoConfig.from_pretrained(
                source_root, trust_remote_code=True, local_files_only=True
            ).to_dict(),
            where="PrismaSnap live original-source config",
        )
        expected_config = canonical_streamed_model_semantic_config(
            identity.get("config"), where="PrismaSnap identity source config"
        )
    except Exception as exc:
        raise RuntimeError("PrismaSnap original-source config cannot be validated") from exc
    if before_config != (config_path.stat().st_size, _sha256_file(config_path)):
        raise RuntimeError("PrismaSnap original-source config changed during validation")
    if live_config != expected_config:
        raise RuntimeError("PrismaSnap original-source config semantics differ")

    rows = identity.get("shards")
    assert isinstance(rows, list)  # validated by validate_streamed_model_identity
    by_name: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise RuntimeError("PrismaSnap original-source shard identity is malformed")
        name = Path(str(row["path"])).name
        if name in by_name:
            raise RuntimeError("PrismaSnap original-source shard identity repeats a name")
        by_name[name] = row
    expected_shards = set(weight_map.values())
    if set(by_name) != expected_shards:
        raise RuntimeError("PrismaSnap original-source shard census differs")
    source_entries = [path for path in source_root.iterdir() if path.suffix == ".safetensors"]
    if (
        any(path.is_symlink() or not path.is_file() for path in source_entries)
        or {path.name for path in source_entries} != expected_shards
    ):
        raise RuntimeError("PrismaSnap original-source shard-file census differs")
    for name in sorted(expected_shards):
        path = source_root / name
        before = path.stat()
        row = by_name[name]
        if type(row.get("size")) is not int or row["size"] != before.st_size:
            raise RuntimeError(f"PrismaSnap original-source shard size changed: {name}")
        if _require_sha256(
            row.get("sha256"), where=f"PrismaSnap source identity shard {name}"
        ) != _sha256_file(path):
            raise RuntimeError(f"PrismaSnap original-source shard content changed: {name}")
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"PrismaSnap original-source shard changed while hashing: {name}")
    portable = portable_streamed_model_content_identity(
        identity, where="PrismaSnap attested portable source identity"
    )
    return identity, portable, hashlib.sha256(identity_bytes).hexdigest()


def _argv_flag(argv: Sequence[object], flag: str, *, where: str) -> str:
    values: list[str] = []
    for index, raw in enumerate(argv):
        if not isinstance(raw, str):
            raise RuntimeError(f"{where} launch argv contains a non-string")
        if raw == flag:
            if index + 1 >= len(argv) or not isinstance(argv[index + 1], str):
                raise RuntimeError(f"{where} launch flag {flag} has no value")
            values.append(str(argv[index + 1]))
        elif raw.startswith(flag + "="):
            values.append(raw.split("=", 1)[1])
    if len(values) != 1 or not values[0]:
        raise RuntimeError(f"{where} must name {flag} exactly once")
    return values[0]


def _validate_measurement_launch(
    manifest: Mapping[str, object],
    payload: Mapping[str, object],
    *,
    mode: str,
    model: Path,
    output: Path,
    where: str,
    teacher_payload: Path | None = None,
    teacher_meta: Path | None = None,
) -> None:
    argv = manifest.get("launch_argv")
    if not isinstance(argv, list) or not argv:
        raise RuntimeError(f"{where} has no launch argv")
    if (
        _argv_flag(argv, "--mode", where=where) != mode
        or _argv_flag(argv, "--dtype", where=where) != "bfloat16"
        or _argv_flag(argv, "--score-positions", where=where) != "all"
        or int(_argv_flag(argv, "--n-samples", where=where))
        != payload.get("n_samples")
        or int(_argv_flag(argv, "--seqlen", where=where)) != payload.get("seqlen")
        or int(_argv_flag(argv, "--prompt-top-k", where=where))
        != payload.get("prompt_top_k")
    ):
        raise RuntimeError(f"{where} launch workload differs from its result")
    _same_resolved_path(_argv_flag(argv, "--model", where=where), model, where=f"{where} argv model")
    _same_resolved_path(_argv_flag(argv, "--output", where=where), output, where=f"{where} argv output")
    forbidden = {
        "--quantization",
        "--allow-spec-decode",
        "--speculative-config",
        "--dsv4-gridbook-contract",
    }
    if any(
        isinstance(item, str)
        and (item in forbidden or any(item.startswith(flag + "=") for flag in forbidden))
        for item in argv
    ):
        raise RuntimeError(f"{where} is not an unmodified BF16 launch")
    if (
        manifest.get("quantization") is not None
        or manifest.get("speculative_config") is not None
        or payload.get("spec_decode_detected") is not False
    ):
        raise RuntimeError(f"{where} BF16/no-spec serve contract failed")
    _same_resolved_path(manifest.get("model"), model, where=f"{where} manifest model")
    if mode == "teacher":
        if teacher_meta is None:
            raise AssertionError("teacher metadata path is required")
        _same_resolved_path(
            _argv_flag(argv, "--meta-output", where=where),
            teacher_meta,
            where=f"{where} argv metadata output",
        )
    else:
        if teacher_payload is None:
            raise AssertionError("student teacher payload path is required")
        _same_resolved_path(
            _argv_flag(argv, "--teacher-payload", where=where),
            teacher_payload,
            where=f"{where} argv teacher payload",
        )


def _validated_serve_attestation(
    payload: Mapping[str, object], *, where: str
) -> tuple[Mapping[str, object], str]:
    manifest = payload.get("serve_manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"{where} lacks an embedded serve manifest")
    if manifest.get("measurement_tool") != "measure_vllm_full_kl":
        raise RuntimeError(f"{where} names the wrong measurement tool")
    producer = manifest.get("producer_identity")
    if (
        not isinstance(producer, Mapping)
        or set(producer)
        != {
            "schema",
            "measurement_tool",
            "git_commit",
            "git_tree",
            "git_dirty",
            "source_files",
            "source_files_sha256",
        }
        or producer.get("schema") != "prismaquant.gold_producer_identity/1"
        or producer.get("measurement_tool") != "measure_vllm_full_kl"
        or producer.get("git_dirty") is not False
        or not isinstance(producer.get("source_files"), Mapping)
        or not isinstance(producer.get("source_files_sha256"), str)
        or _SHA256.fullmatch(str(producer["source_files_sha256"])) is None
    ):
        raise RuntimeError(f"{where} has malformed gold-producer identity")
    if (
        not isinstance(producer.get("git_commit"), str)
        or re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})", str(producer["git_commit"])
        )
        is None
        or (
            producer.get("git_tree") is not None
            and (
                not isinstance(producer.get("git_tree"), str)
                or re.fullmatch(r"[0-9a-f]{40}", str(producer["git_tree"])) is None
            )
        )
    ):
        raise RuntimeError(f"{where} gold-producer git identity is malformed")
    source_files = producer["source_files"]
    assert isinstance(source_files, Mapping)
    if not source_files:
        raise RuntimeError(f"{where} gold-producer source closure is empty")
    for name, raw in source_files.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"{where} gold-producer source path is malformed")
        row = _require_exact_mapping(
            raw, {"bytes", "sha256"}, where=f"{where} producer source {name}"
        )
        _require_positive_int(
            row.get("bytes"), where=f"{where} producer source bytes"
        )
        _require_sha256(
            row.get("sha256"), where=f"{where} producer source digest"
        )
    if producer.get("source_files_sha256") != canonical_json_sha256(
        source_files, where="PrismaQuant gold producer source files"
    ):
        raise RuntimeError(f"{where} gold-producer source closure digest differs")
    try:
        from tools.serve_fingerprint import fingerprint, performance_stack_fingerprint

        expected_performance = performance_stack_fingerprint(manifest)
        expected_serve = fingerprint(manifest)
    except Exception as exc:
        raise RuntimeError(f"{where} serve manifest cannot be replayed") from exc
    recorded_performance = manifest.get("performance_stack_fingerprint")
    recorded_serve = manifest.get("serve_fingerprint")
    if (
        recorded_performance != expected_performance
        or recorded_serve != expected_serve
        or payload.get("serve_fingerprint") != expected_serve
    ):
        raise RuntimeError(f"{where} serve fingerprint is stale or inconsistent")
    top_performance = payload.get("performance_stack_fingerprint")
    if top_performance is not None and top_performance != expected_performance:
        raise RuntimeError(f"{where} performance-stack fingerprint differs")
    return manifest, expected_serve


def _calibration_tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().to(device="cpu").contiguous()
    raw = contiguous.numpy().tobytes(order="C")
    return canonical_json_sha256(
        {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        },
        where="PrismaSnap fold calibration ids",
    )


def _validated_teacher_payload(
    data: bytes,
    *,
    teacher_meta: Mapping[str, object],
    source_root: Path,
) -> tuple[dict[str, object], str]:
    # This is the same trusted local torch payload consumed by the gold KL
    # evaluator.  Read once, then both decode and hash these exact bytes.
    try:
        raw = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("PrismaSnap teacher payload is unreadable") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("PrismaSnap teacher payload is not a mapping")
    required = {
        "score_positions", "prompt_top_k", "topk_ids", "topk_lps",
        "calib_ids", "starts", "model", "n_samples", "seqlen", "vocab_size",
    }
    if set(raw) != required or raw.get("score_positions") != "all":
        raise RuntimeError("PrismaSnap teacher payload schema is malformed")
    n = raw.get("n_samples")
    seqlen = raw.get("seqlen")
    top_k = raw.get("prompt_top_k")
    vocab = raw.get("vocab_size")
    if any(type(value) is not int or int(value) <= 0 for value in (n, seqlen, top_k, vocab)):
        raise RuntimeError("PrismaSnap teacher payload dimensions are malformed")
    n_i, seq_i, top_i, vocab_i = int(n), int(seqlen), int(top_k), int(vocab)
    ids = raw.get("topk_ids")
    lps = raw.get("topk_lps")
    calib = raw.get("calib_ids")
    if (
        not isinstance(ids, torch.Tensor)
        or not isinstance(lps, torch.Tensor)
        or not isinstance(calib, torch.Tensor)
        or tuple(ids.shape) != (n_i, seq_i - 1, top_i)
        or tuple(lps.shape) != tuple(ids.shape)
        or tuple(calib.shape) != (n_i, seq_i)
        or ids.dtype not in {torch.int32, torch.int64}
        or calib.dtype not in {torch.int32, torch.int64}
        or lps.dtype not in {torch.float32, torch.float64}
        or bool((calib < 0).any().item())
        or bool((calib >= vocab_i).any().item())
    ):
        raise RuntimeError("PrismaSnap teacher payload tensor contract failed")
    valid = ids >= 0
    padding = ids == -1
    valid_lps = torch.isfinite(lps) & (lps <= 0.0)
    if (
        bool((~(valid | padding)).any().item())
        or bool((valid & (ids >= vocab_i)).any().item())
        or bool((valid & ~valid_lps).any().item())
        or bool((padding & ~torch.isneginf(lps)).any().item())
        or bool((~padding & torch.isneginf(lps)).any().item())
        or bool((valid.sum(dim=-1) < 1).any().item())
        or bool((lps[..., 1:] > lps[..., :-1]).any().item())
    ):
        raise RuntimeError("PrismaSnap teacher top-K support is malformed")
    sortable = torch.where(valid, ids.to(torch.int64), vocab_i)
    sorted_ids = torch.sort(sortable, dim=-1).values
    if bool(
        (
            (sorted_ids[..., 1:] == sorted_ids[..., :-1])
            & (sorted_ids[..., 1:] < vocab_i)
        ).any().item()
    ):
        raise RuntimeError("PrismaSnap teacher top-K support repeats a token")
    probability_mass = torch.where(valid, lps, -torch.inf).to(torch.float64).exp().sum(dim=-1)
    if (
        not bool(torch.isfinite(probability_mass).all().item())
        or bool((probability_mass <= 0.0).any().item())
        or bool((probability_mass > 1.0 + 1e-5).any().item())
    ):
        raise RuntimeError("PrismaSnap teacher top-K probability mass is invalid")
    starts = raw.get("starts")
    if (
        not isinstance(starts, list)
        or len(starts) != n_i
        or any(type(value) is not int or value < 0 for value in starts)
        or teacher_meta.get("starts") != starts
    ):
        raise RuntimeError("PrismaSnap teacher calibration windows differ")
    _same_resolved_path(raw.get("model"), source_root, where="teacher payload model")
    for key in ("score_positions", "prompt_top_k", "n_samples", "seqlen", "vocab_size"):
        if raw.get(key) != teacher_meta.get(key):
            raise RuntimeError("PrismaSnap teacher payload metadata differs")
    shape = teacher_meta.get("teacher_shape")
    if shape != list(lps.shape):
        raise RuntimeError("PrismaSnap teacher shape receipt differs from payload")
    return dict(raw), _calibration_tensor_sha256(calib)


def _validate_student_metrics(student: Mapping[str, object]) -> None:
    n = student.get("n_samples")
    seqlen = student.get("seqlen")
    if type(n) is not int or type(seqlen) is not int:
        raise RuntimeError("PrismaSnap student dimensions are malformed")
    if student.get("n_positions") != n * (seqlen - 1):
        raise RuntimeError("PrismaSnap student scored-position census differs")
    per_sample = student.get("kl_per_sample")
    if (
        not isinstance(per_sample, list)
        or len(per_sample) != n
        or any(not _is_number(value) or not math.isfinite(float(value)) or value < 0 for value in per_sample)
    ):
        raise RuntimeError("PrismaSnap student per-sample KL is malformed")
    kl_mean = student.get("kl_mean")
    if not _is_number(kl_mean) or not math.isclose(
        sum(float(value) for value in per_sample) / n,
        float(kl_mean),
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise RuntimeError("PrismaSnap student KL mean is incoherent")
    for key in ("kl_p99", "kl_max"):
        value = student.get(key)
        if not _is_number(value) or not math.isfinite(float(value)) or float(value) < 0:
            raise RuntimeError(f"PrismaSnap student {key} is malformed")
    if float(student["kl_max"]) < float(student["kl_p99"]):
        raise RuntimeError("PrismaSnap student KL tail metrics are incoherent")


def attest_fold_fidelity(
    checkpoint_dir: str | Path,
    student_result_path: str | Path,
    teacher_meta_path: str | Path,
    source_identity_path: str | Path,
) -> dict[str, object]:
    """Attest a standard all-position BF16-vs-BF16 served KL result.

    The student result and teacher metadata are content-hashed, path-bound to
    the materialized checkpoint/original source, and required to describe the
    same teacher payload and calibration windows.  The checkpoint's current
    shard bytes are re-hashed before the state transition.
    """
    root = Path(checkpoint_dir)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"PrismaSnap checkpoint is not a real directory: {root}")
    root = root.resolve(strict=True)
    requested_result = Path(student_result_path)
    requested_meta = Path(teacher_meta_path)
    requested_source_identity = Path(source_identity_path)
    student_bytes = _read_regular_bytes(
        requested_result, where="PrismaSnap student KL result"
    )
    teacher_meta_bytes = _read_regular_bytes(
        requested_meta, where="PrismaSnap teacher metadata"
    )
    result_path = requested_result.resolve(strict=True)
    meta_path = requested_meta.resolve(strict=True)
    identity_path = requested_source_identity.resolve(strict=True)
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        provenance_path = root / PROVENANCE_JSON
        provenance = _load_json(provenance_path, where="PrismaSnap provenance")
        validate_prismasnap_provenance_payload(
            provenance, require_verified=False, where="PrismaSnap provenance"
        )
        existing_fold = provenance.get("fold_fidelity")
        materialized_digest = (
            str(existing_fold.get("materialized_provenance_sha256"))
            if provenance.get("state") == "VERIFIED"
            and isinstance(existing_fold, Mapping)
            else str(provenance["provenance_sha256"])
        )
        student = _json_from_bytes(
            student_bytes, where="PrismaSnap student KL result"
        )
        teacher = _json_from_bytes(
            teacher_meta_bytes, where="PrismaSnap teacher metadata"
        )

        if student.get("mode") != "student" or teacher.get("mode") != "teacher":
            raise RuntimeError("PrismaSnap fold evidence has wrong evaluator modes")
        if student.get("quantization") is not None:
            raise RuntimeError("PrismaSnap fold fidelity must serve the BF16 checkpoint")
        if student.get("score_positions") != "all" or teacher.get("score_positions") != "all":
            raise RuntimeError("PrismaSnap fold fidelity must score all positions")
        _same_resolved_path(student.get("model"), root, where="student result model")
        manifest, student_serve_fingerprint = _validated_serve_attestation(
            student, where="PrismaSnap student result"
        )
        if manifest.get("quantization") is not None:
            raise RuntimeError("PrismaSnap student result lacks a BF16 serve manifest")
        _same_resolved_path(manifest.get("model"), root, where="student serve manifest")

        source_model = provenance.get("source_model")
        if not isinstance(source_model, str):
            raise RuntimeError("PrismaSnap provenance lacks its original source path")
        source_root = Path(source_model).resolve(strict=True)
        source_identity, portable_source, source_identity_sha256 = (
            _validated_source_identity_binding(source_root, identity_path)
        )
        if (
            source_identity.get("content_sha256")
            != provenance.get("source_local_content_sha256")
            or portable_source.get("portable_content_sha256")
            != provenance.get("source_portable_content_sha256")
        ):
            raise RuntimeError(
                "PrismaSnap original-source identity differs from materialization"
            )
        _same_resolved_path(teacher.get("model"), source_root, where="teacher metadata model")
        teacher_manifest, teacher_serve_fingerprint = _validated_serve_attestation(
            teacher, where="PrismaSnap teacher metadata"
        )
        if teacher_manifest.get("quantization") is not None:
            raise RuntimeError("PrismaSnap teacher metadata is not a BF16 serve")
        _same_resolved_path(
            teacher_manifest.get("model"), source_root, where="teacher serve manifest"
        )
        if student.get("teacher_model") is not None:
            _same_resolved_path(
                student.get("teacher_model"), source_root, where="student teacher model"
            )

        settings = ("score_positions", "prompt_top_k", "n_samples", "seqlen", "vocab_size")
        if any(student.get(key) != teacher.get(key) for key in settings):
            raise RuntimeError("PrismaSnap student/teacher calibration settings differ")
        if (
            type(student.get("n_samples")) is not int
            or int(student["n_samples"]) < 2
            or type(student.get("seqlen")) is not int
            or int(student["seqlen"]) < 512
        ):
            raise RuntimeError("PrismaSnap fold evidence is below the 2x512 minimum")
        _validate_student_metrics(student)

        requested_payload = Path(str(student.get("teacher_payload", "")))
        teacher_payload_bytes = _read_regular_bytes(
            requested_payload, where="PrismaSnap teacher payload"
        )
        teacher_payload_sha256 = hashlib.sha256(teacher_payload_bytes).hexdigest()
        if student.get("teacher_payload_sha256") != teacher_payload_sha256:
            raise RuntimeError(
                "PrismaSnap student result names different teacher payload bytes"
            )
        teacher_payload = requested_payload.resolve(strict=True)
        _same_resolved_path(teacher.get("output"), teacher_payload, where="teacher payload")
        _validate_measurement_launch(
            teacher_manifest,
            teacher,
            mode="teacher",
            model=source_root,
            output=teacher_payload,
            teacher_meta=meta_path,
            where="PrismaSnap teacher metadata",
        )
        _validate_measurement_launch(
            manifest,
            student,
            mode="student",
            model=root,
            output=result_path,
            teacher_payload=teacher_payload,
            teacher_meta=meta_path,
            where="PrismaSnap student result",
        )
        _payload, calibration_ids_sha256 = _validated_teacher_payload(
            teacher_payload_bytes,
            teacher_meta=teacher,
            source_root=source_root,
        )
        corpus = teacher.get("corpus")
        if (
            not isinstance(corpus, Mapping)
            or not isinstance(corpus.get("corpus_sha256"), str)
            or _SHA256.fullmatch(str(corpus["corpus_sha256"])) is None
            or type(corpus.get("total_tokens")) is not int
            or int(corpus["total_tokens"]) <= 0
        ):
            raise RuntimeError("PrismaSnap teacher corpus identity is malformed")
        kl_value = student.get("kl_mean")
        verification = provenance.get("fp64_invariance")
        if not isinstance(verification, Mapping):
            raise RuntimeError("PrismaSnap provenance lacks the numerical threshold")
        limit = verification.get("required_bf16_fold_kl_max")
        if (
            not _is_number(kl_value)
            or not _is_number(limit)
            or not math.isfinite(float(kl_value))
            or float(kl_value) < 0.0
            or float(kl_value) > float(limit)
        ):
            raise RuntimeError(
                f"PrismaSnap BF16 fold KL {kl_value!r} exceeds {limit!r}"
            )

        checkpoint_identity = _checkpoint_content_identity(root)
        output = provenance.get("output")
        if not isinstance(output, Mapping) or any(
            output.get(key) != value for key, value in checkpoint_identity.items()
        ):
            raise RuntimeError("PrismaSnap checkpoint bytes changed before attestation")
        fold: dict[str, object] = {
            "schema": FOLD_FIDELITY_SCHEMA,
            "passed": True,
            "metric": "forward_kl_original_bf16_to_snapped_bf16",
            "kl_mean": float(kl_value),
            "threshold": float(limit),
            "score_positions": student["score_positions"],
            "prompt_top_k": student["prompt_top_k"],
            "n_samples": student["n_samples"],
            "seqlen": student["seqlen"],
            "vocab_size": student["vocab_size"],
            "student_result_sha256": hashlib.sha256(student_bytes).hexdigest(),
            "teacher_meta_sha256": hashlib.sha256(teacher_meta_bytes).hexdigest(),
            "teacher_payload_sha256": teacher_payload_sha256,
            "source_identity_sha256": source_identity_sha256,
            "source_portable_content_sha256": provenance[
                "source_portable_content_sha256"
            ],
            "source_local_content_sha256": provenance[
                "source_local_content_sha256"
            ],
            "calibration_ids_sha256": calibration_ids_sha256,
            "calibration_starts": list(teacher["starts"]),
            "calibration_corpus_sha256": corpus["corpus_sha256"],
            "checkpoint_shard_content_sha256": checkpoint_identity[
                "shard_content_sha256"
            ],
            "checkpoint_weight_map_sha256": checkpoint_identity[
                "checkpoint_weight_map_sha256"
            ],
            "checkpoint_index_sha256": checkpoint_identity["index_sha256"],
            "materialized_provenance_sha256": materialized_digest,
            "serve_fingerprint": student_serve_fingerprint,
            "teacher_serve_fingerprint": teacher_serve_fingerprint,
        }
        if provenance.get("state") == "VERIFIED":
            if provenance.get("fold_fidelity") != fold:
                raise RuntimeError("PrismaSnap checkpoint was verified with other evidence")
            return dict(provenance)
        provenance["state"] = "VERIFIED"
        provenance["fold_fidelity"] = fold
        provenance["provenance_sha256"] = _provenance_digest(provenance)
        validate_prismasnap_provenance_payload(
            provenance, require_verified=True, where="PrismaSnap verified provenance"
        )
        temporary = root / f".{PROVENANCE_JSON}.verification.tmp"
        if os.path.lexists(temporary):
            raise RuntimeError(f"stale PrismaSnap verification temporary: {temporary}")
        data = json.dumps(
            provenance, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, provenance_path)
        os.fsync(directory_fd)
        return dict(provenance)
    finally:
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)
