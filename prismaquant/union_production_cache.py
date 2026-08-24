"""Exact, fail-closed union for striped ``ProductionWeightCache`` builds.

This module is deliberately unrelated to the archived ``smart union``
research path.  It never selects candidates or prunes a format menu.  Its
only operation is the set union of disjoint, already-rendered cache keys.

The portable workflow has three steps::

    python -m prismaquant.union_production_cache manifest ...
    python -m prismaquant.union_production_cache union ...
    python -m prismaquant.union_production_cache verify ...

Each input shard is a self-contained bundle.  Its manifest binds the cache
pickle and every backing tensor by SHA-256, plus value-bearing identities for
the source checkpoint, calibration corpus, producer code, common settings,
and exact render coverage.  Union refuses overlaps, failed renders,
identity drift, missing or modified files, incomplete expected coverage,
and cache metadata it cannot merge without changing semantics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from prismaquant import format_registry as fr
from prismaquant.cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json,
    canonical_json_sha256,
)
from prismaquant.cost_streaming import validate_streamed_model_identity
from prismaquant.layer_config import load_assignment
from prismaquant.nvfp4_cb_footprint import assignment_serialization_sha256
from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    _is_cb_format_name,
    _production_cache_git_commit,
    _production_cache_source_sha256,
    first_identity_difference,
    identity_value_for_error,
)


CAMPAIGN_IDENTITY_SCHEMA = (
    "prismaquant.production_weight_cache.union_campaign_identity.v1"
)
SOURCE_BINDING_SCHEMA = (
    "prismaquant.production_weight_cache.portable_source_binding.v1"
)
RENDER_IDENTITY_SCHEMA = (
    "prismaquant.production_weight_cache.union_render_identity.v1"
)
SHARD_MANIFEST_SCHEMA = (
    "prismaquant.production_weight_cache.shard_manifest.v1"
)
SHARD_MANIFEST_PAYLOAD_SCHEMA = (
    "prismaquant.production_weight_cache.shard_manifest_payload.v1"
)
UNION_MANIFEST_SCHEMA = (
    "prismaquant.production_weight_cache.union_manifest.v1"
)
UNION_MANIFEST_PAYLOAD_SCHEMA = (
    "prismaquant.production_weight_cache.union_manifest_payload.v1"
)
UNION_METADATA_SCHEMA = (
    "prismaquant.production_weight_cache.exact_union.v1"
)
COVERAGE_SCHEMA = (
    "prismaquant.production_weight_cache.union_expected_coverage.v1"
)
MTP_RENDER_METADATA_SCHEMA = (
    "prismaquant.production_weight_cache.mtp_render.v1"
)

_HEX_32_TO_64 = re.compile(r"[0-9a-f]{32,64}")
_FULL_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHARD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")

_REQUIRED_SHARED_METADATA_KEYS = (
    "render_scope",
    "requested_formats",
    "calib_hash",
    "render_mechanism_order",
)
_OPTIONAL_SHARED_METADATA_KEYS = (
    "render_retention",
    "streaming",
    "format_plan_identity_sha256",
)
_MERGED_METADATA_KEYS = frozenset({
    "requested_entries",
    "render_scores",
    "packed_expert_coverage",
    "mtp_render",
    "render_failures",
    "exact_union",
    "render_gates",
    "four_over_six",
    "fisher_weighted_gptq",
})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, where: str) -> str:
    digest = str(value or "").strip().lower()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return digest


def _safe_relative_path(raw: object, *, where: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{where} must be a nonempty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"{where} is not a safe bundle-relative path: {raw!r}")
    return path


def _relative_to_bundle(path: Path, root: Path, *, where: str) -> str:
    resolved_root = root.resolve()
    try:
        relative = path.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{where} must be inside shard bundle {resolved_root}"
        ) from exc
    if relative == Path("."):
        raise ValueError(f"{where} cannot be the shard bundle root")
    return relative.as_posix()


def _canonical_mapping(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{where} must be a nonempty JSON object")
    canonical = canonical_json(value, where=where)
    if not isinstance(canonical, dict):  # defensive; Mapping canonicalizes to dict
        raise ValueError(f"{where} must canonicalize to a JSON object")
    return canonical


def portable_source_binding(source_identity: Mapping[str, object]) -> dict[str, object]:
    """Return a path-independent binding for a validated checkpoint identity.

    ``build_streamed_model_identity`` includes absolute shard paths so its own
    cache can detect replacement in place.  Two hosts may mount the same exact
    checkpoint at different paths, therefore the union binding retains the
    value-bearing config/maps/shard hashes while deliberately excluding only
    location strings.
    """
    validated = validate_streamed_model_identity(
        source_identity, where="ProductionWeightCache union source"
    )
    shards = sorted(
        (
            {
                "size": int(shard["size"]),
                "sha256": _require_sha256(
                    shard["sha256"], where="source shard sha256"
                ),
            }
            for shard in validated["shards"]
        ),
        key=lambda row: (row["sha256"], row["size"]),
    )
    value_bearing: dict[str, object] = {
        "config": validated.get("config"),
        "weight_map": validated.get("weight_map"),
        "shards": shards,
    }
    if "checkpoint_weight_map" in validated:
        value_bearing["checkpoint_weight_map"] = validated[
            "checkpoint_weight_map"
        ]
    return {
        "schema": SOURCE_BINDING_SCHEMA,
        "content_sha256": canonical_json_sha256(
            value_bearing, where="portable source checkpoint identity"
        ),
        "config_sha256": canonical_json_sha256(
            value_bearing["config"], where="portable source config"
        ),
        "weight_map_sha256": canonical_json_sha256(
            value_bearing["weight_map"], where="portable source weight map"
        ),
        "checkpoint_weight_map_sha256": (
            canonical_json_sha256(
                value_bearing["checkpoint_weight_map"],
                where="portable source checkpoint weight map",
            )
            if "checkpoint_weight_map" in value_bearing
            else None
        ),
        "shards": shards,
        "shard_count": len(shards),
        "total_bytes": sum(int(row["size"]) for row in shards),
        "resolved_commit": validated.get("resolved_commit"),
    }


def _finalize_coverage(payload: Mapping[str, object]) -> dict[str, object]:
    canonical = _canonical_mapping(payload, where="union expected coverage")
    canonical["identity_sha256"] = canonical_json_sha256(
        canonical, where="union expected coverage identity"
    )
    return canonical


def assignment_coverage(assignment: Mapping[str, str]) -> dict[str, object]:
    pairs = sorted(
        [str(qname), fr.canonical_format_name(str(fmt).strip().upper())]
        for qname, fmt in assignment.items()
        if fr.canonical_format_name(str(fmt).strip().upper()) != "BF16"
    )
    return _finalize_coverage({
        "schema": COVERAGE_SCHEMA,
        "mode": "assignment",
        "qnames": sorted({pair[0] for pair in pairs}),
        "formats": sorted({pair[1] for pair in pairs}),
        "pairs": pairs,
        "assignment_sha256": assignment_serialization_sha256(assignment),
        "stripe_plan_sha256": None,
    })


def format_menu_coverage(
    qnames: Sequence[str],
    formats: Sequence[str],
    *,
    stripe_plan_sha256: str | None = None,
) -> dict[str, object]:
    names = sorted({str(name).strip() for name in qnames if str(name).strip()})
    canonical_formats = sorted({
        fr.canonical_format_name(str(fmt).strip().upper())
        for fmt in formats
        if str(fmt).strip()
        and fr.canonical_format_name(str(fmt).strip().upper()) != "BF16"
    })
    if not names or not canonical_formats:
        raise ValueError("format-menu coverage requires qnames and formats")
    return _finalize_coverage({
        "schema": COVERAGE_SCHEMA,
        "mode": "format-menu",
        "qnames": names,
        "formats": canonical_formats,
        "pairs": [[name, fmt] for name in names for fmt in canonical_formats],
        "assignment_sha256": None,
        "stripe_plan_sha256": stripe_plan_sha256,
    })


def validate_expected_coverage(value: object) -> dict[str, object]:
    raw = _canonical_mapping(value, where="union expected coverage")
    if raw.get("schema") != COVERAGE_SCHEMA:
        raise ValueError("unsupported union expected-coverage schema")
    mode = raw.get("mode")
    if mode not in {"assignment", "format-menu"}:
        raise ValueError("union expected coverage has unsupported mode")
    qnames = raw.get("qnames")
    formats = raw.get("formats")
    pairs = raw.get("pairs")
    if not isinstance(qnames, list) or not qnames or qnames != sorted(set(qnames)):
        raise ValueError("union expected coverage qnames are not canonical")
    if not isinstance(formats, list) or not formats or formats != sorted(set(formats)):
        raise ValueError("union expected coverage formats are not canonical")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("union expected coverage pairs are malformed")
    canonical_pairs: list[list[str]] = []
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(item, str) and item for item in pair)
        ):
            raise ValueError("union expected coverage contains malformed pair")
        qname, fmt = pair
        canonical_fmt = fr.canonical_format_name(fmt.strip().upper())
        if canonical_fmt != fmt or fmt == "BF16":
            raise ValueError("union expected coverage contains noncanonical format")
        canonical_pairs.append([qname, fmt])
    if canonical_pairs != sorted(canonical_pairs) or len(canonical_pairs) != len({
        tuple(pair) for pair in canonical_pairs
    }):
        raise ValueError("union expected coverage pairs are not unique and sorted")
    if sorted({pair[0] for pair in canonical_pairs}) != qnames:
        raise ValueError("union expected coverage qname projection differs")
    if sorted({pair[1] for pair in canonical_pairs}) != formats:
        raise ValueError("union expected coverage format projection differs")
    if mode == "format-menu" and canonical_pairs != [
        [name, fmt] for name in qnames for fmt in formats
    ]:
        raise ValueError("format-menu coverage is not the exact Cartesian product")
    stored = _require_sha256(
        raw.get("identity_sha256"), where="expected coverage identity_sha256"
    )
    payload = dict(raw)
    payload.pop("identity_sha256", None)
    if stored != canonical_json_sha256(
        payload, where="union expected coverage identity"
    ):
        raise ValueError("union expected coverage identity SHA-256 mismatch")
    return raw


def load_stripe_plan_coverage(path: str | Path) -> dict[str, object]:
    """Load and verify the planner's files into exact format-menu coverage."""
    plan_path = Path(path)
    raw = _read_json_object(plan_path, where="production-cache stripe plan")
    from prismaquant.production_cache_stripes import SCHEMA as STRIPE_PLAN_SCHEMA

    if raw.get("schema") != STRIPE_PLAN_SCHEMA:
        raise ValueError("unsupported production-cache stripe-plan schema")
    formats = raw.get("formats")
    stripes = raw.get("stripes")
    if (
        not isinstance(formats, list)
        or not formats
        or not isinstance(stripes, list)
        or not stripes
    ):
        raise ValueError("production-cache stripe plan is malformed")
    canonical_formats = [
        fr.canonical_format_name(str(fmt).strip().upper()) for fmt in formats
    ]
    if (
        any(fmt == "BF16" for fmt in canonical_formats)
        or len(canonical_formats) != len(set(canonical_formats))
        or any(raw != canonical for raw, canonical in zip(
            formats, canonical_formats, strict=True
        ))
    ):
        raise ValueError("production-cache stripe plan formats are not canonical")
    declared_stripes = raw.get("n_stripes")
    if (
        not isinstance(declared_stripes, int)
        or declared_stripes != len(stripes)
        or declared_stripes <= 0
    ):
        raise ValueError("production-cache stripe count is inconsistent")
    qnames: list[str] = []
    seen_indices: set[int] = set()
    for row in stripes:
        if not isinstance(row, Mapping):
            raise ValueError("production-cache stripe row is malformed")
        index = row.get("index")
        if not isinstance(index, int) or index in seen_indices:
            raise ValueError("production-cache stripe indices are invalid")
        seen_indices.add(index)
        qname_path = plan_path.parent / _safe_relative_path(
            row.get("path"), where=f"stripe {index} qname path"
        )
        if qname_path.is_symlink() or not qname_path.is_file():
            raise ValueError(f"stripe {index} qname file is missing or unsafe")
        expected_digest = _require_sha256(
            row.get("sha256"), where=f"stripe {index} qname file sha256"
        )
        if _file_sha256(qname_path) != expected_digest:
            raise ValueError(f"stripe {index} qname file SHA-256 mismatch")
        names = [
            line.strip()
            for line in qname_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if names != sorted(set(names)):
            raise ValueError(f"stripe {index} qnames are not unique and sorted")
        if len(names) != int(row.get("qnames", -1)):
            raise ValueError(f"stripe {index} qname count mismatch")
        overlap = sorted(set(qnames) & set(names))
        if overlap:
            raise ValueError(
                f"stripe plan qnames overlap; sample={overlap[:8]}"
            )
        qnames.extend(names)
    if len(qnames) != int(raw.get("qnames", -1)):
        raise ValueError("stripe plan total qname count mismatch")
    if seen_indices != set(range(declared_stripes)):
        raise ValueError("production-cache stripe indices are not contiguous")
    semantic_plan = {
        "schema": raw["schema"],
        "profile": raw.get("profile"),
        "probe_sha256": raw.get("probe_sha256"),
        "formats": formats,
        "stripes": [
            {
                "index": row["index"],
                "sha256": row["sha256"],
                "qnames": row["qnames"],
                "groups": row.get("groups"),
                "estimated_work": row.get("estimated_work"),
                "parameters": row.get("parameters"),
            }
            for row in sorted(stripes, key=lambda item: item["index"])
        ],
    }
    return format_menu_coverage(
        qnames,
        canonical_formats,
        stripe_plan_sha256=canonical_json_sha256(
            semantic_plan, where="production-cache stripe plan identity"
        ),
    )


def _coerce_coverage(
    *,
    assignment: Mapping[str, str] | None = None,
    coverage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if (assignment is None) == (coverage is None):
        raise ValueError("provide exactly one of assignment or expected coverage")
    return validate_expected_coverage(
        assignment_coverage(assignment) if assignment is not None else coverage
    )


def _current_code_identity() -> dict[str, str]:
    return {
        "git_commit": _production_cache_git_commit(),
        "producer_source_sha256": _production_cache_source_sha256(),
    }


def _validate_code_identity(value: object, *, where: str) -> dict[str, str]:
    raw = _canonical_mapping(value, where=where)
    commit = str(raw.get("git_commit", "")).lower()
    source = str(raw.get("producer_source_sha256", "")).lower()
    if _FULL_OBJECT_ID.fullmatch(commit) is None:
        raise ValueError(f"{where}.git_commit must be a full object id")
    _require_sha256(source, where=f"{where}.producer_source_sha256")
    if set(raw) != {"git_commit", "producer_source_sha256"}:
        raise ValueError(f"{where} has unsupported fields")
    return {"git_commit": commit, "producer_source_sha256": source}


def _cache_metadata(cache: ProductionWeightCache) -> dict[str, object]:
    metadata = getattr(cache, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("ProductionWeightCache union requires metadata")
    return dict(metadata)


def _validate_render_scores(
    metadata: Mapping[str, object], *, where: str, expected_entries: int
) -> dict[str, object]:
    raw = metadata.get("render_scores")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} requires metadata.render_scores")
    records = raw.get("records")
    if not isinstance(records, Mapping):
        raise ValueError(f"{where} render_scores.records is malformed")
    declared = raw.get("entries")
    if not isinstance(declared, int) or declared != len(records):
        raise ValueError(
            f"{where} render_scores entry count differs from its records"
        )
    if declared != int(expected_entries):
        raise ValueError(
            f"{where} has {expected_entries} cache entries but {declared} "
            "render-score records"
        )
    return canonical_json(raw, where=f"{where} render scores")


def _render_identity(
    cache: ProductionWeightCache,
    coverage: Mapping[str, object],
) -> dict[str, object]:
    coverage = validate_expected_coverage(coverage)
    metadata = _cache_metadata(cache)
    render_scope = metadata.get("render_scope")
    if render_scope != coverage["mode"]:
        raise ValueError(
            "cache render_scope differs from expected coverage mode: "
            f"{render_scope!r} != {coverage['mode']!r}"
        )
    render_retention = metadata.get("render_retention", "materialized")
    if render_retention != "materialized":
        raise ValueError("exact cache union requires materialized render retention")
    streaming = metadata.get("streaming", False)
    if not isinstance(streaming, bool):
        raise ValueError("cache streaming metadata must be boolean when present")
    calibration_hash = str(metadata.get("calib_hash", "")).strip().lower()
    if _HEX_32_TO_64.fullmatch(calibration_hash) is None:
        raise ValueError("exact cache union requires a value-bearing calib_hash")
    requested_formats = metadata.get("requested_formats")
    if (
        not isinstance(requested_formats, Sequence)
        or isinstance(requested_formats, (str, bytes))
        or not requested_formats
    ):
        raise ValueError("exact cache union requires requested_formats")
    formats = sorted({
        fr.canonical_format_name(str(fmt).strip().upper())
        for fmt in requested_formats
    })
    if formats != coverage["formats"]:
        raise ValueError(
            "cache requested_formats differ from exact expected coverage"
        )
    mechanism_order = metadata.get("render_mechanism_order")
    if (
        not isinstance(mechanism_order, Sequence)
        or isinstance(mechanism_order, (str, bytes))
        or not mechanism_order
    ):
        raise ValueError("exact cache union requires render_mechanism_order")
    declared_entries = metadata.get("requested_entries")
    if not isinstance(declared_entries, int) or declared_entries != len(cache):
        raise ValueError(
            "cache requested_entries must equal its materialized key count"
        )
    _validate_render_scores(
        metadata, where="ProductionWeightCache shard", expected_entries=len(cache)
    )
    return {
        "schema": RENDER_IDENTITY_SCHEMA,
        "coverage_identity_sha256": coverage["identity_sha256"],
        "coverage_mode": coverage["mode"],
        "assignment_sha256": coverage.get("assignment_sha256"),
        "stripe_plan_sha256": coverage.get("stripe_plan_sha256"),
        "levers": canonical_json(cache.levers, where="cache render levers"),
        "render_scope": render_scope,
        "render_retention": "materialized",
        "streaming": streaming,
        "requested_formats": formats,
        "format_plan_identity_sha256": metadata.get(
            "format_plan_identity_sha256"
        ),
        "render_mechanism_order": canonical_json(
            mechanism_order, where="cache render mechanism order"
        ),
    }


def build_campaign_identity(
    cache: ProductionWeightCache,
    *,
    source_model_identity: Mapping[str, object],
    settings: Mapping[str, object],
    assignment: Mapping[str, str] | None = None,
    coverage: Mapping[str, object] | None = None,
    code_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the shared identity every striped render must match exactly."""
    expected = _coerce_coverage(assignment=assignment, coverage=coverage)
    settings_payload = _canonical_mapping(settings, where="union settings")
    metadata = _cache_metadata(cache)
    calibration_hash = str(metadata.get("calib_hash", "")).strip().lower()
    if _HEX_32_TO_64.fullmatch(calibration_hash) is None:
        raise ValueError("union campaign requires a value-bearing calib_hash")
    payload: dict[str, object] = {
        "schema": CAMPAIGN_IDENTITY_SCHEMA,
        "source": portable_source_binding(source_model_identity),
        "calibration": {
            "data_hash": calibration_hash,
        },
        "code": _validate_code_identity(
            code_identity or _current_code_identity(), where="union code identity"
        ),
        "settings": {
            "payload": settings_payload,
            "sha256": canonical_json_sha256(
                settings_payload, where="union settings identity"
            ),
        },
        "render": _render_identity(cache, expected),
    }
    payload["identity_sha256"] = canonical_json_sha256(
        payload, where="ProductionWeightCache union campaign identity"
    )
    return payload


def validate_campaign_identity(value: object) -> dict[str, object]:
    raw = _canonical_mapping(value, where="union campaign identity")
    if raw.get("schema") != CAMPAIGN_IDENTITY_SCHEMA:
        raise ValueError("unsupported union campaign identity schema")
    stored_digest = _require_sha256(
        raw.get("identity_sha256"), where="campaign identity_sha256"
    )
    digest_payload = dict(raw)
    digest_payload.pop("identity_sha256", None)
    expected_digest = canonical_json_sha256(
        digest_payload, where="union campaign identity payload"
    )
    if stored_digest != expected_digest:
        raise ValueError("union campaign identity_sha256 does not match payload")
    source = _canonical_mapping(raw.get("source"), where="campaign source")
    if source.get("schema") != SOURCE_BINDING_SCHEMA:
        raise ValueError("unsupported portable source binding schema")
    _require_sha256(source.get("content_sha256"), where="source content_sha256")
    calibration = _canonical_mapping(
        raw.get("calibration"), where="campaign calibration"
    )
    if _HEX_32_TO_64.fullmatch(
        str(calibration.get("data_hash", "")).lower()
    ) is None:
        raise ValueError("campaign calibration data_hash is malformed")
    _validate_code_identity(raw.get("code"), where="campaign code")
    settings = _canonical_mapping(raw.get("settings"), where="campaign settings")
    settings_payload = _canonical_mapping(
        settings.get("payload"), where="campaign settings payload"
    )
    if _require_sha256(
        settings.get("sha256"), where="campaign settings sha256"
    ) != canonical_json_sha256(
        settings_payload, where="campaign settings payload"
    ):
        raise ValueError("campaign settings sha256 does not match payload")
    render = _canonical_mapping(raw.get("render"), where="campaign render")
    if render.get("schema") != RENDER_IDENTITY_SCHEMA:
        raise ValueError("unsupported union render identity schema")
    _require_sha256(
        render.get("coverage_identity_sha256"),
        where="campaign coverage_identity_sha256",
    )
    if render.get("coverage_mode") not in {"assignment", "format-menu"}:
        raise ValueError("campaign render coverage mode is unsupported")
    if render.get("coverage_mode") == "assignment":
        _require_sha256(
            render.get("assignment_sha256"),
            where="campaign assignment_sha256",
        )
    elif render.get("assignment_sha256") is not None:
        raise ValueError("format-menu campaign cannot carry assignment identity")
    return raw


def _assert_identity_equal(
    reference: object, candidate: object, *, where: str
) -> None:
    difference = first_identity_difference(reference, candidate)
    if difference is None:
        return
    field, stored, current = difference
    raise ValueError(
        f"{where} identity differs at {field!r}: "
        f"reference={identity_value_for_error(stored)} "
        f"candidate={identity_value_for_error(current)}"
    )


def _load_cache(path: Path) -> ProductionWeightCache:
    try:
        with path.open("rb") as handle:
            cache = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"cannot load ProductionWeightCache {path}") from exc
    if not isinstance(cache, ProductionWeightCache):
        raise TypeError(f"{path} does not contain a ProductionWeightCache")
    return cache


def _expected_keys_for_coverage(
    cache: ProductionWeightCache,
    coverage: Mapping[str, object],
    *,
    require_complete: bool,
) -> set[tuple[str, str]]:
    expected_coverage = validate_expected_coverage(coverage)
    expected = {
        (str(pair[0]), str(pair[1])) for pair in expected_coverage["pairs"]
    }
    stored = set(cache.weights)
    extras = sorted(stored - expected)
    coverage_label = (
        "assignment" if expected_coverage["mode"] == "assignment"
        else "expected format-menu coverage"
    )
    if extras:
        raise ValueError(
            "ProductionWeightCache shard contains keys outside the exact "
            f"{coverage_label}; sample={extras[:8]}"
        )
    missing = sorted(expected - stored)
    if require_complete and missing:
        raise ValueError(
            f"ProductionWeightCache union is missing {coverage_label} entries: "
            f"{len(missing)} sample={missing[:8]}"
        )
    return expected


def _backing_records(
    cache: ProductionWeightCache,
    *,
    cache_dir: Path,
) -> list[dict[str, object]]:
    if cache.failed:
        raise ValueError(
            f"ProductionWeightCache has {len(cache.failed)} failed renders"
        )
    if not cache.weights:
        raise ValueError("ProductionWeightCache shard is empty")
    resolved_root = cache_dir.resolve()
    if not resolved_root.is_dir() or cache_dir.is_symlink():
        raise ValueError(f"cache directory is missing or unsafe: {cache_dir}")
    records: list[dict[str, object]] = []
    seen_paths: dict[Path, tuple[str, str]] = {}
    for key, value in sorted(cache.weights.items()):
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(item, str) and item for item in key)
        ):
            raise ValueError(f"malformed ProductionWeightCache key {key!r}")
        qname, fmt = key
        canonical_fmt = fr.canonical_format_name(fmt.strip().upper())
        if canonical_fmt != fmt:
            raise ValueError(
                f"cache key {key!r} does not use its canonical format name"
            )
        if _is_cb_format_name(fmt):
            raise ValueError(
                "exact cache union does not yet merge CB pair identities; "
                "refusing rather than dropping their integrity metadata"
            )
        if isinstance(value, torch.Tensor):
            raise ValueError(
                f"cache key {key!r} is in-memory; shard union requires a "
                "portable disk-backed cache"
            )
        raw_path = Path(str(value))
        path = raw_path if raw_path.is_absolute() else resolved_root / raw_path
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"cache key {key!r} escapes cache directory: {path}"
            ) from exc
        if path.is_symlink() or not resolved.is_file():
            raise ValueError(
                f"cache key {key!r} backing file is missing or unsafe: {path}"
            )
        prior = seen_paths.get(resolved)
        if prior is not None:
            raise ValueError(
                f"cache keys {prior!r} and {key!r} share backing file {path}"
            )
        seen_paths[resolved] = key
        stat = resolved.stat()
        records.append({
            "qname": qname,
            "format": fmt,
            "path": relative.as_posix(),
            "size": int(stat.st_size),
            "sha256": _file_sha256(resolved),
        })
    return records


def _manifest_envelope(
    *, schema: str, payload: Mapping[str, object], where: str
) -> dict[str, object]:
    canonical_payload = _canonical_mapping(payload, where=where)
    return {
        "schema": schema,
        "payload": canonical_payload,
        "payload_sha256": canonical_json_sha256(
            canonical_payload, where=f"{where} digest"
        ),
    }


def create_shard_manifest(
    *,
    cache_path: str | Path,
    cache_dir: str | Path,
    manifest_path: str | Path,
    shard_id: str,
    source_model_identity: Mapping[str, object],
    settings: Mapping[str, object],
    assignment: Mapping[str, str] | None = None,
    coverage: Mapping[str, object] | None = None,
    code_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create and durably publish a portable manifest for one cache stripe."""
    expected = _coerce_coverage(assignment=assignment, coverage=coverage)
    if _SHARD_ID.fullmatch(str(shard_id)) is None:
        raise ValueError(f"invalid cache shard id {shard_id!r}")
    cache_path = Path(cache_path)
    cache_dir = Path(cache_dir)
    manifest_path = Path(manifest_path)
    bundle_root = manifest_path.parent
    bundle_root.mkdir(parents=True, exist_ok=True)
    cache_relative = _relative_to_bundle(
        cache_path, bundle_root, where="cache pickle"
    )
    cache_dir_relative = _relative_to_bundle(
        cache_dir, bundle_root, where="cache directory"
    )
    if cache_path.is_symlink() or not cache_path.is_file():
        raise ValueError(f"cache pickle is missing or unsafe: {cache_path}")
    cache = _load_cache(cache_path)
    cache.relocate(cache_dir)
    _expected_keys_for_coverage(cache, expected, require_complete=False)
    records = _backing_records(cache, cache_dir=cache_dir)
    campaign = build_campaign_identity(
        cache,
        source_model_identity=source_model_identity,
        settings=settings,
        coverage=expected,
        code_identity=code_identity,
    )
    payload = {
        "schema": SHARD_MANIFEST_PAYLOAD_SCHEMA,
        "shard_id": str(shard_id),
        "campaign_identity": campaign,
        "coverage_identity_sha256": expected["identity_sha256"],
        "cache_pickle": {
            "path": cache_relative,
            "size": int(cache_path.stat().st_size),
            "sha256": _file_sha256(cache_path),
        },
        "cache_dir": cache_dir_relative,
        "entries": len(cache),
        "keys_sha256": canonical_json_sha256(
            [[record["qname"], record["format"]] for record in records],
            where="cache shard keys",
        ),
        "backing_files": records,
        "cache_metadata_sha256": canonical_json_sha256(
            cache.metadata, where="cache shard metadata"
        ),
        "activation_max_abs_sha256": canonical_json_sha256(
            cache.activation_max_abs or {}, where="cache activation maxima"
        ),
    }
    envelope = _manifest_envelope(
        schema=SHARD_MANIFEST_SCHEMA,
        payload=payload,
        where="ProductionWeightCache shard manifest",
    )
    atomic_write_bytes(
        manifest_path,
        json.dumps(
            envelope,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8"),
    )
    return envelope


def _load_manifest_envelope(
    path: Path, *, schema: str, payload_schema: str, where: str
) -> tuple[dict[str, object], str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read {where} {path}") from exc
    envelope = _canonical_mapping(raw, where=where)
    if envelope.get("schema") != schema:
        raise ValueError(f"{where} has unsupported schema")
    payload = _canonical_mapping(envelope.get("payload"), where=f"{where} payload")
    if payload.get("schema") != payload_schema:
        raise ValueError(f"{where} payload has unsupported schema")
    stored = _require_sha256(
        envelope.get("payload_sha256"), where=f"{where} payload_sha256"
    )
    observed = canonical_json_sha256(payload, where=f"{where} payload")
    if stored != observed:
        raise ValueError(f"{where} payload SHA-256 mismatch")
    return payload, stored


def _verify_record_set(
    *,
    cache: ProductionWeightCache,
    cache_dir: Path,
    stored_records: object,
    where: str,
) -> list[dict[str, object]]:
    if not isinstance(stored_records, list):
        raise ValueError(f"{where} backing_files is malformed")
    observed = _backing_records(cache, cache_dir=cache_dir)
    expected = canonical_json(stored_records, where=f"{where} backing files")
    _assert_identity_equal(expected, observed, where=f"{where} backing-file")
    return observed


def verify_shard_manifest(
    manifest_path: str | Path,
    *,
    assignment: Mapping[str, str] | None = None,
    coverage: Mapping[str, object] | None = None,
    require_current_code: bool = True,
) -> tuple[dict[str, object], ProductionWeightCache, list[dict[str, object]]]:
    """Verify one transferred shard bundle before it is admitted to union."""
    expected = _coerce_coverage(assignment=assignment, coverage=coverage)
    path = Path(manifest_path)
    payload, _ = _load_manifest_envelope(
        path,
        schema=SHARD_MANIFEST_SCHEMA,
        payload_schema=SHARD_MANIFEST_PAYLOAD_SCHEMA,
        where="ProductionWeightCache shard manifest",
    )
    if _SHARD_ID.fullmatch(str(payload.get("shard_id", ""))) is None:
        raise ValueError("shard manifest has invalid shard_id")
    campaign = validate_campaign_identity(payload.get("campaign_identity"))
    if payload.get("coverage_identity_sha256") != expected["identity_sha256"]:
        raise ValueError("shard manifest expected-coverage identity mismatch")
    if campaign["render"]["coverage_identity_sha256"] != expected[
        "identity_sha256"
    ]:
        raise ValueError("shard campaign expected-coverage identity mismatch")
    if require_current_code:
        _assert_identity_equal(
            campaign["code"],
            _validate_code_identity(
                _current_code_identity(), where="current union code"
            ),
            where="current producer code",
        )
    root = path.parent.resolve()
    cache_record = _canonical_mapping(
        payload.get("cache_pickle"), where="shard cache pickle record"
    )
    cache_path = root / _safe_relative_path(
        cache_record.get("path"), where="shard cache pickle path"
    )
    cache_dir = root / _safe_relative_path(
        payload.get("cache_dir"), where="shard cache directory"
    )
    if cache_path.is_symlink() or not cache_path.is_file():
        raise ValueError(f"shard cache pickle is missing or unsafe: {cache_path}")
    if int(cache_record.get("size", -1)) != cache_path.stat().st_size:
        raise ValueError("shard cache pickle size mismatch")
    if _require_sha256(
        cache_record.get("sha256"), where="shard cache pickle sha256"
    ) != _file_sha256(cache_path):
        raise ValueError("shard cache pickle SHA-256 mismatch")
    cache = _load_cache(cache_path)
    cache.relocate(cache_dir)
    if payload.get("entries") != len(cache):
        raise ValueError("shard manifest entry count differs from cache")
    _expected_keys_for_coverage(cache, expected, require_complete=False)
    records = _verify_record_set(
        cache=cache,
        cache_dir=cache_dir,
        stored_records=payload.get("backing_files"),
        where=f"cache shard {payload['shard_id']}",
    )
    keys_digest = canonical_json_sha256(
        [[record["qname"], record["format"]] for record in records],
        where="verified cache shard keys",
    )
    if _require_sha256(
        payload.get("keys_sha256"), where="shard keys_sha256"
    ) != keys_digest:
        raise ValueError("shard key-set SHA-256 mismatch")
    metadata_digest = canonical_json_sha256(
        cache.metadata, where="verified shard metadata"
    )
    if _require_sha256(
        payload.get("cache_metadata_sha256"), where="shard metadata sha256"
    ) != metadata_digest:
        raise ValueError("shard cache metadata SHA-256 mismatch")
    activation_digest = canonical_json_sha256(
        cache.activation_max_abs or {}, where="verified activation maxima"
    )
    if _require_sha256(
        payload.get("activation_max_abs_sha256"),
        where="shard activation maxima sha256",
    ) != activation_digest:
        raise ValueError("shard activation-maxima SHA-256 mismatch")
    derived_render = _render_identity(cache, expected)
    _assert_identity_equal(
        campaign["render"], derived_render, where="shard render"
    )
    if campaign["calibration"]["data_hash"] != cache.metadata["calib_hash"]:
        raise ValueError("shard calibration identity differs from cache metadata")
    return payload, cache, records


def _merge_disjoint_mapping(
    destination: dict[str, object],
    source: Mapping[str, object],
    *,
    where: str,
) -> None:
    overlap = sorted(set(destination) & set(source))
    if overlap:
        raise ValueError(f"{where} overlaps across shards; sample={overlap[:8]}")
    destination.update(copy.deepcopy(dict(source)))


def _merge_mtp_render(caches: Sequence[ProductionWeightCache]) -> dict | None:
    rows = [
        cache.metadata.get("mtp_render")
        for cache in caches
        if isinstance(cache.metadata, Mapping)
        and cache.metadata.get("mtp_render") is not None
    ]
    if not rows:
        return None
    qnames: set[str] = set()
    formats_by_qname: dict[str, object] = {}
    activation_rows: dict[str, object] = {}
    common: dict[str, object] | None = None
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"MTP render metadata shard {index} is malformed")
        expected_fields = {
            "schema", "scope", "entries", "qnames", "formats",
            "formats_by_qname", "source_prefix", "source_tensor_count",
            "activation_rows", "max_act_rows",
        }
        if set(raw) != expected_fields:
            raise ValueError("MTP render metadata has unsupported/missing fields")
        if raw.get("schema") != MTP_RENDER_METADATA_SCHEMA:
            raise ValueError("MTP render metadata schema is unsupported")
        if raw.get("scope") not in {"assignment", "format-menu"}:
            raise ValueError("MTP render scope is unsupported")
        if not isinstance(raw.get("source_prefix"), str) or not raw.get(
            "source_prefix"
        ):
            raise ValueError("MTP source_prefix is malformed")
        if not isinstance(raw.get("source_tensor_count"), int) or int(
            raw["source_tensor_count"]
        ) <= 0:
            raise ValueError("MTP source_tensor_count is malformed")
        if not isinstance(raw.get("max_act_rows"), int) or int(
            raw["max_act_rows"]
        ) <= 0:
            raise ValueError("MTP max_act_rows is malformed")
        raw_qnames = raw.get("qnames")
        raw_formats = raw.get("formats")
        raw_formats_by_qname = raw.get("formats_by_qname")
        raw_activation_rows = raw.get("activation_rows")
        if (
            not isinstance(raw_qnames, Sequence)
            or isinstance(raw_qnames, (str, bytes))
            or not isinstance(raw_formats, Sequence)
            or isinstance(raw_formats, (str, bytes))
            or not isinstance(raw_formats_by_qname, Mapping)
            or not isinstance(raw_activation_rows, Mapping)
        ):
            raise ValueError("MTP render coverage metadata is malformed")
        names = [str(name) for name in raw_qnames]
        if names != sorted(set(names)):
            raise ValueError("MTP render qnames are not canonical")
        if set(raw_formats_by_qname) != set(names):
            raise ValueError("MTP formats_by_qname differs from qnames")
        if set(raw_activation_rows) != set(names):
            raise ValueError("MTP activation_rows differs from qnames")
        normalized_formats: dict[str, list[str]] = {}
        for name in names:
            values = raw_formats_by_qname[name]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError("MTP formats_by_qname row is malformed")
            raw_normalized = [
                fr.canonical_format_name(str(fmt).strip().upper())
                for fmt in values
            ]
            if (
                not raw_normalized
                or "BF16" in raw_normalized
                or len(raw_normalized) != len(set(raw_normalized))
            ):
                raise ValueError("MTP formats_by_qname row is malformed")
            normalized = sorted(raw_normalized)
            normalized_formats[name] = normalized
            rows_value = raw_activation_rows[name]
            if not isinstance(rows_value, int) or rows_value <= 0:
                raise ValueError("MTP activation_rows value is malformed")
        declared_formats_raw = [
            fr.canonical_format_name(str(fmt).strip().upper())
            for fmt in raw_formats
        ]
        if len(declared_formats_raw) != len(set(declared_formats_raw)):
            raise ValueError("MTP formats contain duplicates")
        declared_formats = sorted(declared_formats_raw)
        projected_formats = sorted({
            fmt for values in normalized_formats.values() for fmt in values
        })
        if declared_formats != projected_formats:
            raise ValueError("MTP formats differ from formats_by_qname")
        declared_entries = raw.get("entries")
        expected_entries = sum(len(values) for values in normalized_formats.values())
        if declared_entries != expected_entries:
            raise ValueError("MTP render entry count differs from pair coverage")
        current_common = {
            key: copy.deepcopy(raw.get(key))
            for key in (
                "schema",
                "scope",
                "source_prefix",
                "source_tensor_count",
                "max_act_rows",
            )
        }
        if common is None:
            common = current_common
        else:
            _assert_identity_equal(
                common, current_common, where="MTP render common metadata"
            )
        overlap = sorted(qnames & set(names))
        if overlap:
            raise ValueError(
                f"MTP render qnames overlap across shards; sample={overlap[:8]}"
            )
        qnames.update(names)
        formats_by_qname.update(normalized_formats)
        activation_rows.update({
            name: int(raw_activation_rows[name]) for name in names
        })
    assert common is not None
    formats = sorted({
        fmt for values in formats_by_qname.values() for fmt in values
    })
    return {
        **common,
        "entries": sum(len(values) for values in formats_by_qname.values()),
        "qnames": sorted(qnames),
        "formats": formats,
        "formats_by_qname": dict(sorted(formats_by_qname.items())),
        "activation_rows": dict(sorted(activation_rows.items())),
    }


def _merge_counter_buckets(
    values: Sequence[Mapping[str, object]], *, where: str
) -> dict[str, object]:
    expected = {"accepted", "rejected", "package_accepted", "reasons"}
    totals = {"accepted": 0, "rejected": 0, "package_accepted": 0}
    reasons: dict[str, int] = {}
    for value in values:
        if set(value) != expected or not isinstance(value.get("reasons"), Mapping):
            raise ValueError(f"{where} counter metadata is malformed")
        for key in totals:
            count = value.get(key)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{where}.{key} counter is malformed")
            totals[key] += count
        for reason, count in value["reasons"].items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{where}.reasons counter is malformed")
            name = str(reason)
            reasons[name] = reasons.get(name, 0) + count
    return {**totals, "reasons": dict(sorted(reasons.items()))}


def _merge_mechanism_summaries(
    values: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    names = sorted(set().union(*(value.keys() for value in values)))
    merged: dict[str, object] = {}
    for name in names:
        buckets = [value[name] for value in values if name in value]
        if any(not isinstance(bucket, Mapping) for bucket in buckets):
            raise ValueError("render_gates mechanism bucket is malformed")
        merged[str(name)] = _merge_counter_buckets(
            buckets, where=f"render_gates.mechanisms.{name}"
        )
    return merged


def _merge_render_gates(metadata: Sequence[Mapping[str, object]]) -> dict | None:
    rows = [item.get("render_gates") for item in metadata]
    if all(row is None for row in rows):
        return None
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("render_gates must be present on every shard")
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    mechanism_rows = [row.get("mechanisms") for row in rows]
    if any(not isinstance(value, Mapping) for value in mechanism_rows):
        raise ValueError("render_gates.mechanisms is malformed")
    mechanisms = _merge_mechanism_summaries(mechanism_rows)
    enabled = rows[0].get("enabled")
    for index, row in enumerate(rows):
        if row.get("enabled") is not enabled:
            raise ValueError("render_gates.enabled differs across shards")
        current = row.get("records")
        if not isinstance(current, list) or row.get("entries") != len(current):
            raise ValueError(f"render_gates shard {index} records are malformed")
        for record in current:
            if not isinstance(record, Mapping):
                raise ValueError("render_gates record is malformed")
            pair = (str(record.get("qname", "")), str(record.get("format", "")))
            if not all(pair) or pair in seen:
                raise ValueError("render_gates records overlap or lack pair identity")
            seen.add(pair)
            records.append(copy.deepcopy(dict(record)))
    records.sort(key=lambda row: (str(row["qname"]), str(row["format"])))
    return {
        "enabled": enabled,
        "entries": len(records),
        "mechanisms": mechanisms,
        "records": records,
    }


def _merge_fisher_metadata(metadata: Sequence[Mapping[str, object]]) -> dict | None:
    rows = [item.get("fisher_weighted_gptq") for item in metadata]
    if all(row is None for row in rows):
        return None
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("fisher_weighted_gptq must be present on every shard")
    required = {"enabled", "h_detail_dir", "loaded", "misses"}
    if any(set(row) != required for row in rows):
        raise ValueError("fisher_weighted_gptq metadata is malformed")
    for key in ("enabled", "h_detail_dir"):
        for row in rows[1:]:
            _assert_identity_equal(rows[0][key], row[key], where=f"fisher {key}")
    for row in rows:
        for key in ("loaded", "misses"):
            if (
                not isinstance(row[key], int)
                or isinstance(row[key], bool)
                or int(row[key]) < 0
            ):
                raise ValueError(f"fisher_weighted_gptq.{key} is malformed")
    return {
        "enabled": copy.deepcopy(rows[0]["enabled"]),
        "h_detail_dir": copy.deepcopy(rows[0]["h_detail_dir"]),
        "loaded": sum(int(row["loaded"]) for row in rows),
        "misses": sum(int(row["misses"]) for row in rows),
    }


def _merge_cache_metadata(
    caches: Sequence[ProductionWeightCache],
    *,
    campaign: Mapping[str, object],
    shard_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not caches:
        raise ValueError("cannot merge zero ProductionWeightCaches")
    metadata = [_cache_metadata(cache) for cache in caches]
    reference: dict[str, object] = {}
    for key in _REQUIRED_SHARED_METADATA_KEYS:
        if key not in metadata[0]:
            raise ValueError(f"cache metadata is missing shared field {key!r}")
        reference[key] = copy.deepcopy(metadata[0][key])
        for index, candidate in enumerate(metadata[1:], start=1):
            if key not in candidate:
                raise ValueError(
                    f"cache shard {index} is missing shared metadata {key!r}"
                )
            _assert_identity_equal(
                reference[key], candidate[key], where=f"cache metadata {key}"
            )
    optional_defaults = {
        "render_retention": "materialized",
        "streaming": False,
        "format_plan_identity_sha256": None,
    }
    for key in _OPTIONAL_SHARED_METADATA_KEYS:
        value = metadata[0].get(key, optional_defaults[key])
        for candidate in metadata[1:]:
            _assert_identity_equal(
                value,
                candidate.get(key, optional_defaults[key]),
                where=f"cache metadata {key}",
            )
        reference[key] = copy.deepcopy(value)

    # Unknown metadata may be retained only when it is exactly shared.  A
    # differing field needs an explicit semantic merge rule; silently choosing
    # one shard would make the union dependent on input order.
    all_keys = set().union(*(item.keys() for item in metadata))
    shared_keys = set(_REQUIRED_SHARED_METADATA_KEYS) | set(
        _OPTIONAL_SHARED_METADATA_KEYS
    )
    unmanaged = sorted(all_keys - shared_keys - _MERGED_METADATA_KEYS)
    for key in unmanaged:
        if key not in metadata[0]:
            raise ValueError(
                f"cache metadata {key!r} exists only on a non-reference shard"
            )
        value = metadata[0][key]
        for index, candidate in enumerate(metadata[1:], start=1):
            if key not in candidate:
                raise ValueError(
                    f"cache shard {index} is missing metadata {key!r}"
                )
            _assert_identity_equal(value, candidate[key], where=f"cache metadata {key}")
        reference[key] = copy.deepcopy(value)

    render_records: dict[str, object] = {}
    render_common = None
    for index, (cache, item) in enumerate(zip(caches, metadata, strict=True)):
        scores = _validate_render_scores(
            item, where=f"cache shard {index}", expected_entries=len(cache)
        )
        current_common = {
            key: copy.deepcopy(value)
            for key, value in scores.items()
            if key not in {"entries", "records"}
        }
        if render_common is None:
            render_common = current_common
        else:
            _assert_identity_equal(
                render_common, current_common, where="render-score metadata"
            )
        _merge_disjoint_mapping(
            render_records,
            scores["records"],
            where="render-score records",
        )
        failures = item.get("render_failures")
        if failures not in (None, {}):
            raise ValueError(f"cache shard {index} records render failures")

    expert_coverage: dict[str, object] = {}
    for index, item in enumerate(metadata):
        coverage = item.get("packed_expert_coverage")
        if coverage is None:
            continue
        if not isinstance(coverage, Mapping):
            raise ValueError(
                f"cache shard {index} packed_expert_coverage is malformed"
            )
        _merge_disjoint_mapping(
            expert_coverage,
            coverage,
            where="packed-expert coverage",
        )

    entries = sum(len(cache) for cache in caches)
    if len(render_records) != entries:
        raise ValueError(
            "merged render-score coverage differs from merged cache keys"
        )
    reference["requested_entries"] = entries
    reference["render_scores"] = {
        **(render_common or {}),
        "entries": len(render_records),
        "records": dict(sorted(render_records.items())),
    }
    if expert_coverage:
        reference["packed_expert_coverage"] = dict(sorted(expert_coverage.items()))
    mtp_render = _merge_mtp_render(caches)
    if mtp_render is not None:
        if mtp_render["scope"] != campaign["render"]["render_scope"]:
            raise ValueError("MTP render scope differs from campaign render scope")
        reference["mtp_render"] = mtp_render
    render_gates = _merge_render_gates(metadata)
    if render_gates is not None:
        expected_gate_entries = entries - (
            int(mtp_render["entries"]) if mtp_render is not None else 0
        )
        if render_gates["entries"] != expected_gate_entries:
            raise ValueError("merged render_gates coverage differs from cache keys")
        reference["render_gates"] = render_gates
    four_over_six = [item.get("four_over_six") for item in metadata]
    if any(value is not None for value in four_over_six):
        if any(not isinstance(value, Mapping) for value in four_over_six):
            raise ValueError("four_over_six must be present on every shard")
        reference["four_over_six"] = _merge_counter_buckets(
            four_over_six, where="four_over_six"
        )
    fisher = _merge_fisher_metadata(metadata)
    if fisher is not None:
        reference["fisher_weighted_gptq"] = fisher
    reference["exact_union"] = {
        "schema": UNION_METADATA_SCHEMA,
        "campaign_identity_sha256": campaign["identity_sha256"],
        "coverage_identity_sha256": campaign["render"][
            "coverage_identity_sha256"
        ],
        "coverage_mode": campaign["render"]["coverage_mode"],
        "assignment_sha256": campaign["render"].get("assignment_sha256"),
        "stripe_plan_sha256": campaign["render"].get("stripe_plan_sha256"),
        "entries": entries,
        "input_shards": [
            {
                "shard_id": receipt["shard_id"],
                "manifest_payload_sha256": receipt["manifest_payload_sha256"],
            }
            for receipt in sorted(shard_receipts, key=lambda row: row["shard_id"])
        ],
    }
    return dict(sorted(reference.items()))


def _copy_verified(source: Path, destination: Path, *, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    observed = _file_sha256(destination)
    if observed != expected_sha256:
        raise ValueError(
            f"copied cache shard SHA-256 mismatch for {destination.name}"
        )


def _union_weight_filename(key: tuple[str, str], content_sha256: str) -> str:
    key_sha256 = canonical_json_sha256(
        [key[0], key[1]], where="union cache backing key"
    )
    return f"{key_sha256}-{content_sha256}.pt"


def _union_backing_records(
    cache: ProductionWeightCache, *, cache_dir: Path
) -> list[dict[str, object]]:
    return _backing_records(cache, cache_dir=cache_dir)


def union_shard_manifests(
    manifest_paths: Sequence[str | Path],
    *,
    assignment: Mapping[str, str] | None = None,
    coverage: Mapping[str, object] | None = None,
    output_bundle: str | Path,
    require_current_code: bool = True,
) -> dict[str, object]:
    """Union verified disjoint shards into one atomic, portable cache bundle."""
    expected = _coerce_coverage(assignment=assignment, coverage=coverage)
    if len(manifest_paths) < 2:
        raise ValueError("exact cache union requires at least two shard manifests")
    output_bundle = Path(output_bundle)
    if output_bundle.exists():
        raise FileExistsError(
            f"refusing to overwrite existing union bundle {output_bundle}"
        )
    output_bundle.parent.mkdir(parents=True, exist_ok=True)

    verified = [
        verify_shard_manifest(
            path,
            coverage=expected,
            require_current_code=require_current_code,
        )
        for path in manifest_paths
    ]
    payloads = [item[0] for item in verified]
    caches = [item[1] for item in verified]
    records_by_shard = [item[2] for item in verified]
    shard_ids = [str(payload["shard_id"]) for payload in payloads]
    if len(set(shard_ids)) != len(shard_ids):
        raise ValueError(f"duplicate cache shard ids: {shard_ids}")
    campaign = validate_campaign_identity(payloads[0]["campaign_identity"])
    for index, payload in enumerate(payloads[1:], start=1):
        _assert_identity_equal(
            campaign,
            validate_campaign_identity(payload["campaign_identity"]),
            where=f"cache shard {index} campaign",
        )

    weights: dict[tuple[str, str], object] = {}
    activation_max_abs: dict[str, float] = {}
    source_by_key: dict[tuple[str, str], tuple[Path, dict[str, object]]] = {}
    shard_receipts: list[dict[str, object]] = []
    for manifest_path, payload, cache, records in zip(
        manifest_paths, payloads, caches, records_by_shard, strict=True
    ):
        overlap = sorted(set(weights) & set(cache.weights))
        if overlap:
            raise ValueError(
                "ProductionWeightCache shards overlap; "
                f"sample={overlap[:8]}"
            )
        root = Path(manifest_path).parent.resolve()
        cache_dir = root / _safe_relative_path(
            payload["cache_dir"], where="shard cache directory"
        )
        record_by_key = {
            (str(record["qname"]), str(record["format"])): record
            for record in records
        }
        for key in sorted(cache.weights):
            record = record_by_key[key]
            source = cache_dir / _safe_relative_path(
                record["path"], where=f"backing file for {key!r}"
            )
            weights[key] = _union_weight_filename(key, str(record["sha256"]))
            source_by_key[key] = (source, record)
        maxima = cache.activation_max_abs or {}
        overlap_maxima = sorted(set(activation_max_abs) & set(maxima))
        if overlap_maxima:
            raise ValueError(
                "activation-maxima qnames overlap across shards; "
                f"sample={overlap_maxima[:8]}"
            )
        activation_max_abs.update({
            str(name): float(value) for name, value in maxima.items()
        })
        _, payload_sha = _load_manifest_envelope(
            Path(manifest_path),
            schema=SHARD_MANIFEST_SCHEMA,
            payload_schema=SHARD_MANIFEST_PAYLOAD_SCHEMA,
            where="ProductionWeightCache shard manifest",
        )
        shard_receipts.append({
            "shard_id": payload["shard_id"],
            "manifest_payload_sha256": payload_sha,
        })

    metadata = _merge_cache_metadata(
        caches,
        campaign=campaign,
        shard_receipts=shard_receipts,
    )
    union_cache = ProductionWeightCache(
        weights=dict(sorted(weights.items())),
        levers=copy.deepcopy(caches[0].levers),
        activation_max_abs=dict(sorted(activation_max_abs.items())),
        failed={},
        cache_dir="weights",
        metadata=metadata,
    )
    _expected_keys_for_coverage(union_cache, expected, require_complete=True)

    temporary_root = Path(tempfile.mkdtemp(
        prefix=f".{output_bundle.name}.tmp-", dir=output_bundle.parent
    ))
    try:
        weights_dir = temporary_root / "weights"
        weights_dir.mkdir()
        copied: dict[str, str] = {}
        for key in sorted(source_by_key):
            source, record = source_by_key[key]
            digest = _require_sha256(
                record["sha256"], where=f"backing file {key!r} sha256"
            )
            filename = _union_weight_filename(key, digest)
            destination = weights_dir / filename
            prior = copied.get(filename)
            if prior is None:
                _copy_verified(source, destination, expected_sha256=digest)
                copied[filename] = digest
            elif prior != digest:
                raise ValueError(f"content-addressed filename collision: {filename}")

        cache_bytes = pickle.dumps(union_cache, protocol=pickle.HIGHEST_PROTOCOL)
        cache_path = temporary_root / "production_weight_cache.pkl"
        atomic_write_bytes(cache_path, cache_bytes)
        union_cache.relocate(weights_dir)
        backing_records = _union_backing_records(
            union_cache, cache_dir=weights_dir
        )
        union_payload = {
            "schema": UNION_MANIFEST_PAYLOAD_SCHEMA,
            "campaign_identity": campaign,
            "coverage_identity_sha256": expected["identity_sha256"],
            "coverage_mode": expected["mode"],
            "assignment_sha256": expected.get("assignment_sha256"),
            "stripe_plan_sha256": expected.get("stripe_plan_sha256"),
            "input_shards": sorted(
                shard_receipts, key=lambda row: row["shard_id"]
            ),
            "cache_pickle": {
                "path": "production_weight_cache.pkl",
                "size": int(cache_path.stat().st_size),
                "sha256": _file_sha256(cache_path),
            },
            "cache_dir": "weights",
            "entries": len(union_cache),
            "keys_sha256": canonical_json_sha256(
                [[record["qname"], record["format"]]
                 for record in backing_records],
                where="union cache keys",
            ),
            "backing_files": backing_records,
            "cache_metadata_sha256": canonical_json_sha256(
                union_cache.metadata, where="union cache metadata"
            ),
            "activation_max_abs_sha256": canonical_json_sha256(
                union_cache.activation_max_abs or {},
                where="union activation maxima",
            ),
        }
        envelope = _manifest_envelope(
            schema=UNION_MANIFEST_SCHEMA,
            payload=union_payload,
            where="ProductionWeightCache union manifest",
        )
        manifest_path = temporary_root / "union_manifest.json"
        atomic_write_bytes(
            manifest_path,
            json.dumps(
                envelope,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )
        verify_union_manifest(
            manifest_path,
            coverage=expected,
            require_current_code=require_current_code,
        )
        os.replace(temporary_root, output_bundle)
        parent_fd = os.open(output_bundle.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return envelope
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def verify_union_manifest(
    manifest_path: str | Path,
    *,
    assignment: Mapping[str, str] | None = None,
    coverage: Mapping[str, object] | None = None,
    require_current_code: bool = True,
) -> tuple[dict[str, object], ProductionWeightCache]:
    """Verify a canonical union bundle after publication or transfer."""
    expected = _coerce_coverage(assignment=assignment, coverage=coverage)
    path = Path(manifest_path)
    payload, _ = _load_manifest_envelope(
        path,
        schema=UNION_MANIFEST_SCHEMA,
        payload_schema=UNION_MANIFEST_PAYLOAD_SCHEMA,
        where="ProductionWeightCache union manifest",
    )
    campaign = validate_campaign_identity(payload.get("campaign_identity"))
    if require_current_code:
        _assert_identity_equal(
            campaign["code"],
            _validate_code_identity(
                _current_code_identity(), where="current union code"
            ),
            where="current producer code",
        )
    coverage_digest = expected["identity_sha256"]
    if payload.get("coverage_identity_sha256") != coverage_digest:
        raise ValueError("union manifest expected-coverage identity mismatch")
    if campaign["render"]["coverage_identity_sha256"] != coverage_digest:
        raise ValueError("union campaign render coverage identity mismatch")
    if payload.get("coverage_mode") != expected["mode"]:
        raise ValueError("union manifest coverage mode mismatch")
    if payload.get("assignment_sha256") != expected.get("assignment_sha256"):
        raise ValueError("union manifest assignment identity mismatch")
    if payload.get("stripe_plan_sha256") != expected.get("stripe_plan_sha256"):
        raise ValueError("union manifest stripe-plan identity mismatch")
    root = path.parent.resolve()
    cache_record = _canonical_mapping(
        payload.get("cache_pickle"), where="union cache pickle record"
    )
    cache_path = root / _safe_relative_path(
        cache_record.get("path"), where="union cache pickle path"
    )
    cache_dir = root / _safe_relative_path(
        payload.get("cache_dir"), where="union cache directory"
    )
    if cache_path.is_symlink() or not cache_path.is_file():
        raise ValueError("union cache pickle is missing or unsafe")
    if cache_path.stat().st_size != int(cache_record.get("size", -1)):
        raise ValueError("union cache pickle size mismatch")
    if _file_sha256(cache_path) != _require_sha256(
        cache_record.get("sha256"), where="union cache pickle sha256"
    ):
        raise ValueError("union cache pickle SHA-256 mismatch")
    cache = _load_cache(cache_path)
    cache.relocate(cache_dir)
    if payload.get("entries") != len(cache):
        raise ValueError("union manifest entry count differs from cache")
    _expected_keys_for_coverage(cache, expected, require_complete=True)
    records = _verify_record_set(
        cache=cache,
        cache_dir=cache_dir,
        stored_records=payload.get("backing_files"),
        where="union cache",
    )
    keys_digest = canonical_json_sha256(
        [[record["qname"], record["format"]] for record in records],
        where="verified union cache keys",
    )
    if payload.get("keys_sha256") != keys_digest:
        raise ValueError("union cache key-set SHA-256 mismatch")
    if payload.get("cache_metadata_sha256") != canonical_json_sha256(
        cache.metadata, where="verified union cache metadata"
    ):
        raise ValueError("union cache metadata SHA-256 mismatch")
    if payload.get("activation_max_abs_sha256") != canonical_json_sha256(
        cache.activation_max_abs or {}, where="verified union activation maxima"
    ):
        raise ValueError("union cache activation-maxima SHA-256 mismatch")
    exact_union = (
        cache.metadata.get("exact_union")
        if isinstance(cache.metadata, Mapping)
        else None
    )
    if not isinstance(exact_union, Mapping):
        raise ValueError("union cache is missing exact-union metadata")
    if (
        exact_union.get("schema") != UNION_METADATA_SCHEMA
        or exact_union.get("campaign_identity_sha256")
        != campaign["identity_sha256"]
        or exact_union.get("coverage_identity_sha256") != coverage_digest
        or exact_union.get("coverage_mode") != expected["mode"]
        or exact_union.get("assignment_sha256")
        != expected.get("assignment_sha256")
        or exact_union.get("stripe_plan_sha256")
        != expected.get("stripe_plan_sha256")
        or exact_union.get("entries") != len(cache)
    ):
        raise ValueError("union cache exact-union metadata is inconsistent")
    return payload, cache


def _read_json_object(path: str | Path, *, where: str) -> dict[str, object]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read {where} from {path}") from exc
    return _canonical_mapping(raw, where=where)


def _summary(payload: Mapping[str, object]) -> str:
    render = payload.get("campaign_identity", {}).get("render", {})
    return json.dumps({
        "schema": payload.get("schema"),
        "entries": payload.get("entries"),
        "coverage_mode": payload.get("coverage_mode") or render.get(
            "coverage_mode"
        ),
        "coverage_identity_sha256": (
            payload.get("coverage_identity_sha256")
            or render.get("coverage_identity_sha256")
        ),
        "shard_id": payload.get("shard_id"),
    }, sort_keys=True)


def _add_coverage_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--assignment")
    group.add_argument("--stripe-plan")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact disjoint ProductionWeightCache shard union"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="publish one shard manifest")
    manifest.add_argument("--cache", required=True)
    manifest.add_argument("--cache-dir", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--shard-id", required=True)
    manifest.add_argument("--source-model-identity", required=True)
    manifest.add_argument("--settings", required=True)
    _add_coverage_arguments(manifest)

    verify_shard = sub.add_parser(
        "verify-shard", help="verify one transferred shard bundle"
    )
    verify_shard.add_argument("--manifest", required=True)
    _add_coverage_arguments(verify_shard)

    union = sub.add_parser("union", help="build one canonical exact union bundle")
    union.add_argument("--manifest", action="append", required=True)
    _add_coverage_arguments(union)
    union.add_argument("--output-dir", required=True)

    verify = sub.add_parser("verify", help="verify a canonical union bundle")
    verify.add_argument("--manifest", required=True)
    _add_coverage_arguments(verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    expected = (
        assignment_coverage(load_assignment(args.assignment))
        if args.assignment
        else load_stripe_plan_coverage(args.stripe_plan)
    )
    if args.command == "manifest":
        envelope = create_shard_manifest(
            cache_path=args.cache,
            cache_dir=args.cache_dir,
            manifest_path=args.output,
            shard_id=args.shard_id,
            source_model_identity=_read_json_object(
                args.source_model_identity, where="source model identity"
            ),
            settings=_read_json_object(args.settings, where="union settings"),
            coverage=expected,
        )
        print(_summary(envelope["payload"]))
        return 0
    if args.command == "verify-shard":
        payload, _cache, _records = verify_shard_manifest(
            args.manifest, coverage=expected
        )
        print(_summary(payload))
        return 0
    if args.command == "union":
        envelope = union_shard_manifests(
            args.manifest,
            coverage=expected,
            output_bundle=args.output_dir,
        )
        print(_summary(envelope["payload"]))
        return 0
    payload, _cache = verify_union_manifest(
        args.manifest, coverage=expected
    )
    print(_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
