"""Strict sample-parallel union for probe statistics and activation rows.

Ordinary incremental-probe shards partition Linears and therefore reject
overlapping ``stats`` keys.  Sample-parallel shards do the opposite: every
worker measures the same Linear universe on a disjoint calibration interval.
Their raw Fisher accumulators are additive, their activation bounds merge by
maximum, and their already-normalized values must be discarded and finalized
once against the global token count.

This module owns no probing or cache mechanism.  It only joins outputs from
the existing incremental probe and writes the ordinary activation-cache blob
shape consumed by ``measure_quant_cost.ActivationIndex``.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.incremental_probe import merge_marginals
from prismaquant.sample_parallel_probe import (
    BODY_STATS_AND_ACTIVATION,
    LM_HEAD_STATS_ONLY,
    MTP_STATS_ONLY,
    SampleParallelProbeError,
    _normalize_partition_contract,
    _strict_json_loads,
    importance_execution_identity_sha256,
    token_rows_per_sample,
    validate_execution_identity,
    validate_global_importance_receipt,
    validate_qname_census,
    validate_run_contract,
)
from prismaquant.sample_parallel_probe_contract import (
    ACTIVATION_CACHE_SHARD_SCHEMA,
    ACTIVATION_PRIORITY_SCHEMA,
    ROW_INDICES_SCOPE,
    activation_priority_group,
    # Backward-compatible public re-export only.  The merger below must not
    # use the production Torch implementation as its verifier.
    activation_row_priorities,
)
from prismaquant.sensitivity_probe import finalize_fisher_stats


SAMPLE_PARALLEL_SHARD_SCHEMA = "prismaquant.sample_parallel_probe.partition.v1"
SAMPLE_PARALLEL_COVER_SCHEMA = "prismaquant.probe.sample_cover.v1"
SAMPLE_PARALLEL_MERGE_SCHEMA = "prismaquant.probe.sample_merge.v1"
IMPORTANCE_NORMALIZATION_SCHEMA = (
    "prismaquant.sample_parallel_probe.global_importance_normalization.v2"
)
ACTIVATION_CACHE_MERGE_SCHEMA = (
    "prismaquant.probe.sample_activation_cache_merge.v1"
)
ACTIVATION_CACHE_MANIFEST = "sample_parallel_merge.json"
_ROW_INDEX_SCOPE = ROW_INDICES_SCOPE
_ACTIVATION_FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")
_ACTIVATION_BLOB_KEYS = frozenset({
    "inputs", "name", "row_indices", "row_priorities",
    "sample_parallel_activation",
})
_ACTIVATION_STAMP_KEYS = frozenset({
    "schema", "priority_schema", "priority_group", "selection",
    "row_indices_scope", "layout", "global_calibration_hash",
    "execution_identity_sha256", "partition_index", "rows_limit",
    "candidate_rows",
})
_ACTIVATION_MERGE_MANIFEST_KEYS = frozenset({
    "schema", "cover", "cover_identity_sha256", "exact_disjoint_cover",
    "exact_global_priority_top_r",
    "local_global_top_r_associativity_verified",
    "independent_numpy_priority_verifier", "priority_schema",
    "supported_activation_layout",
    "routed_packed_activation_caches_rejected",
    "fused_group_alignment_verified", "qnames", "max_rows", "records",
    "activation_qname_manifest_sha256", "source_census_sha256",
    "identity_sha256",
})

_REFERENCE_PRIORITY_KEY_DOMAIN = (
    b"prismaquant.sample-parallel.activation-priority.v2\0"
)
_REFERENCE_U16_MASK = (1 << 16) - 1
_REFERENCE_U32_MASK = (1 << 32) - 1
_REFERENCE_U32_DOMAIN = 1 << 32
_REFERENCE_FMIX32_C1 = 0x85EBCA6B
_REFERENCE_FMIX32_C2 = 0xC2B2AE35
_REFERENCE_CALIBRATION_HASH_RE = re.compile(r"[0-9a-f]{32}")

_SCALAR_SUM_FIELDS = ("h_trace_raw", "h_w2_sum_raw", "n_tokens_seen")
_DERIVED_FIELDS = frozenset({
    "h_trace",
    "h_w2_sum",
    "h_trace_per_expert",
    "h_trace_norm_tokens",
    "route_prob",
})
_DENSE_MARGINAL_FIELDS = frozenset({
    "fisher_row", "fisher_col", "g_sq_sum", "act_sq_sum", "act_absmax",
})
# Keep this qualification identical to ``SensitivityCard.validate``.  The
# trace and marginals reduce the same fp32 ``chunk_h`` values, but they use
# different reduction/addition orders across chunks and workers, so bitwise
# equality is not a valid multi-GPU contract.  One part in 1,000 is the
# repository's established wiring-check tolerance for those fp32 reductions.
_FISHER_MARGINAL_REL_TOL = 1e-3
_PACKED_MARGINAL_FIELDS = frozenset({
    "expert_g_sq_sum", "expert_act_sq_sum", "expert_act_absmax",
    "expert_tokens", "h_trace_per_expert_raw", "h_trace_per_expert",
})
_ROUTED_NULLABLE_MARKERS = frozenset({"router_path", "expert_id", "route_prob"})
_PACKED_PRESENCE_MARKERS = frozenset({
    "num_experts", "expert_axis", "_packed_experts_module", "_packed_param",
    "packed_experts",
})
_ROUTED_PAYLOAD_MAPS = (
    "router_counts", "router_totals", "router_active_counts",
    "expert_route_stats", "expert_info",
)
_PRODUCER_PAYLOAD_KEYS = frozenset({
    "stats", "router_counts", "router_totals", "router_active_counts",
    "expert_route_stats", "expert_info", "meta",
})
_DENSE_STAT_BASE_FIELDS = frozenset({
    "h_trace_raw", "h_w2_sum_raw", "h_trace", "h_w2_sum",
    "h_trace_norm_tokens", "w_max_abs", "w_norm_sq", "n_params",
    "in_features", "out_features", "n_tokens_seen", "route_prob",
    "router_path", "expert_id",
})
_PARTITION_KEYS = frozenset({
    "schema", "global_calibration_hash", "calibration_artifact_sha256",
    "global_samples", "seqlen", "partition_count", "partition_index",
    "sample_indices", "sample_start", "sample_stop",
    "local_calibration_hash", "local_samples", "dataset", "model",
    "calib_seed",
})


class SampleParallelMergeError(ValueError):
    """The declared sample partition or one of its payloads is inconsistent."""


def _reference_activation_priority_key(
    global_calibration_hash: str,
    qname: str,
) -> tuple[int, int]:
    """Independent CPU key derivation for persisted-priority verification."""
    calibration_hash = str(global_calibration_hash)
    if _REFERENCE_CALIBRATION_HASH_RE.fullmatch(calibration_hash) is None:
        raise SampleParallelMergeError(
            "activation priority calibration hash is invalid"
        )
    group_utf8 = activation_priority_group(str(qname)).encode("utf-8")
    if len(group_utf8) > _REFERENCE_U32_MASK:
        raise SampleParallelMergeError(
            "activation priority group name exceeds the uint32 key contract"
        )
    material = (
        _REFERENCE_PRIORITY_KEY_DOMAIN
        + bytes.fromhex(calibration_hash)
        + struct.pack("<I", len(group_utf8))
        + group_utf8
    )
    return struct.unpack(
        "<II", hashlib.blake2b(material, digest_size=8).digest()
    )


def _reference_mullo32_scalar(value: int, constant: int) -> int:
    """Arbitrary-precision scalar low-product reference."""
    x = int(value) & _REFERENCE_U32_MASK
    c = int(constant) & _REFERENCE_U32_MASK
    low = x & _REFERENCE_U16_MASK
    high = x >> 16
    return (
        low * c + ((high * (c & _REFERENCE_U16_MASK)) << 16)
    ) & _REFERENCE_U32_MASK


def _reference_fmix32_scalar(value: int) -> int:
    x = int(value) & _REFERENCE_U32_MASK
    x ^= x >> 16
    x = _reference_mullo32_scalar(x, _REFERENCE_FMIX32_C1)
    x ^= x >> 13
    x = _reference_mullo32_scalar(x, _REFERENCE_FMIX32_C2)
    x ^= x >> 16
    return x & _REFERENCE_U32_MASK


def _reference_activation_priority_scalar(
    global_calibration_hash: str,
    qname: str,
    global_row: int,
) -> int:
    row = int(global_row)
    if row < 0 or row >= _REFERENCE_U32_DOMAIN:
        raise SampleParallelMergeError(
            "global activation row is outside the uint32 priority domain"
        )
    k0, k1 = _reference_activation_priority_key(
        global_calibration_hash, qname
    )
    return _reference_fmix32_scalar(
        _reference_fmix32_scalar(row ^ k0) ^ k1
    )


def _reference_mullo32_numpy(
    values: np.ndarray,
    constant: int,
) -> np.ndarray:
    """Overflow-free int64 low uint32 product using 16-bit limbs."""
    x = np.bitwise_and(values, np.int64(_REFERENCE_U32_MASK))
    low = np.bitwise_and(x, np.int64(_REFERENCE_U16_MASK))
    high = np.right_shift(x, 16)
    # Both products and their sum stay below 2**49.  No NumPy signed
    # overflow or platform unsigned-cast semantics participate.
    return np.bitwise_and(
        low * np.int64(constant)
        + np.left_shift(
            high * np.int64(constant & _REFERENCE_U16_MASK), 16
        ),
        np.int64(_REFERENCE_U32_MASK),
    )


def _reference_fmix32_numpy(values: np.ndarray) -> np.ndarray:
    x = np.bitwise_and(values, np.int64(_REFERENCE_U32_MASK))
    x = np.bitwise_xor(x, np.right_shift(x, 16))
    x = _reference_mullo32_numpy(x, _REFERENCE_FMIX32_C1)
    x = np.bitwise_xor(x, np.right_shift(x, 13))
    x = _reference_mullo32_numpy(x, _REFERENCE_FMIX32_C2)
    x = np.bitwise_xor(x, np.right_shift(x, 16))
    return np.bitwise_and(x, np.int64(_REFERENCE_U32_MASK))


def _reference_activation_priorities_numpy(
    global_calibration_hash: str,
    qname: str,
    global_rows: np.ndarray | Sequence[int],
) -> np.ndarray:
    rows = np.asarray(global_rows, dtype=np.int64)
    if rows.ndim != 1:
        raise SampleParallelMergeError(
            "reference activation rows must be one-dimensional"
        )
    if rows.size and (
        bool(np.any(rows < 0))
        or bool(np.any(rows >= _REFERENCE_U32_DOMAIN))
    ):
        raise SampleParallelMergeError(
            "global activation row is outside the uint32 priority domain"
        )
    k0, k1 = _reference_activation_priority_key(
        global_calibration_hash, qname
    )
    first = _reference_fmix32_numpy(
        np.bitwise_xor(rows, np.int64(k0))
    )
    return _reference_fmix32_numpy(
        np.bitwise_xor(first, np.int64(k1))
    ).astype(np.int64, copy=False)


def _reference_top_r_numpy(
    global_calibration_hash: str,
    qname: str,
    global_rows: np.ndarray | Sequence[int],
    rows_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical exact top-R rows/priorities under the independent PRP."""
    rows = np.asarray(global_rows, dtype=np.int64)
    limit = int(rows_limit)
    if rows.ndim != 1 or limit < 1:
        raise SampleParallelMergeError(
            "reference activation top-R inputs are invalid"
        )
    priorities = _reference_activation_priorities_numpy(
        global_calibration_hash, qname, rows
    )
    if np.unique(rows).size != rows.size:
        raise SampleParallelMergeError(
            "reference activation top-R rows are not unique"
        )
    if np.unique(priorities).size != priorities.size:
        raise SampleParallelMergeError(
            "activation priority PRP produced a collision"
        )
    keep = min(limit, int(rows.size))
    order = np.argsort(priorities, kind="stable")[:keep]
    return rows[order].copy(), priorities[order].copy()


def _reference_top_r_global_domain_numpy(
    global_calibration_hash: str,
    qname: str,
    candidate_rows: int,
    rows_limit: int,
    *,
    chunk_rows: int = 1 << 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact bounded-memory top-R over the closed global row domain.

    The public validator cannot trust the retained union that the producer
    persisted: a re-sealed directory could omit a better row.  Replaying the
    complete ``range(candidate_rows)`` domain closes that hole.  Chunking
    bounds verifier memory while exact top-R associativity preserves the same
    result as one monolithic sort.
    """
    total = int(candidate_rows)
    limit = int(rows_limit)
    chunk = int(chunk_rows)
    if total < 1 or total > _REFERENCE_U32_DOMAIN or limit < 1 or chunk < 1:
        raise SampleParallelMergeError(
            "reference global activation top-R domain is invalid"
        )
    retained_rows = np.empty(0, dtype=np.int64)
    retained_priorities = np.empty(0, dtype=np.int64)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        local_rows = np.arange(start, stop, dtype=np.int64)
        local_rows, local_priorities = _reference_top_r_numpy(
            global_calibration_hash, qname, local_rows, limit
        )
        candidate_rows_np = np.concatenate((retained_rows, local_rows))
        candidate_priorities_np = np.concatenate((
            retained_priorities, local_priorities,
        ))
        keep = min(limit, int(candidate_rows_np.size))
        order = np.argsort(candidate_priorities_np, kind="stable")[:keep]
        retained_rows = candidate_rows_np[order].copy()
        retained_priorities = candidate_priorities_np[order].copy()
    return retained_rows, retained_priorities


def build_sample_parallel_cover(
    shards: Sequence[Mapping[str, object]],
    *,
    execution_identity: Mapping[str, object],
    qname_census: Mapping[str, object],
) -> dict[str, object]:
    """Bind producer partition contracts to one expected probe cover."""
    if not shards:
        raise SampleParallelMergeError("sample-parallel cover has no shards")
    first = shards[0]
    body: dict[str, object] = {
        "schema": SAMPLE_PARALLEL_COVER_SCHEMA,
        "global_calibration_hash": first.get("global_calibration_hash"),
        "calibration_artifact_sha256": first.get(
            "calibration_artifact_sha256"
        ),
        "total_samples": int(first.get("global_samples", 0) or 0),
        "seqlen": int(first.get("seqlen", 0) or 0),
        "partition_count": int(first.get("partition_count", 0) or 0),
        "row_indices_scope": _ROW_INDEX_SCOPE,
        "execution_identity": copy.deepcopy(dict(execution_identity)),
        "qname_census": copy.deepcopy(dict(qname_census)),
        "shards": [copy.deepcopy(dict(shard)) for shard in shards],
    }
    normalized, _ordered = _validate_cover(body, allow_missing_identity=True)
    normalized["identity_sha256"] = canonical_json_sha256(
        normalized, where="sample-parallel probe cover"
    )
    return normalized


def _validate_cover(
    cover: Mapping[str, object],
    *,
    allow_missing_identity: bool = False,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    if not isinstance(cover, Mapping):
        raise SampleParallelMergeError("sample-parallel cover is not a mapping")
    allowed_cover_keys = {
        "schema", "global_calibration_hash", "calibration_artifact_sha256",
        "total_samples", "seqlen", "partition_count", "row_indices_scope",
        "execution_identity", "qname_census", "shards", "identity_sha256",
    }
    allowed_shapes = (
        (allowed_cover_keys, allowed_cover_keys - {"identity_sha256"})
        if allow_missing_identity else (allowed_cover_keys,)
    )
    if set(cover) not in allowed_shapes:
        raise SampleParallelMergeError(
            "sample-parallel cover fields differ from the closed v1 contract"
        )
    if cover.get("schema") != SAMPLE_PARALLEL_COVER_SCHEMA:
        raise SampleParallelMergeError("unsupported sample-parallel cover schema")
    total_samples = int(cover.get("total_samples", 0) or 0)
    seqlen = int(cover.get("seqlen", 0) or 0)
    if (
        type(cover.get("total_samples")) is not int
        or type(cover.get("seqlen")) is not int
        or type(cover.get("partition_count")) is not int
        or type(cover.get("global_calibration_hash")) is not str
        or type(cover.get("calibration_artifact_sha256")) is not str
    ):
        raise SampleParallelMergeError(
            "sample-parallel cover scalar field types differ"
        )
    if total_samples < 1 or seqlen < 1:
        raise SampleParallelMergeError(
            "sample-parallel cover needs positive total_samples and seqlen"
        )
    if cover.get("row_indices_scope") != _ROW_INDEX_SCOPE:
        raise SampleParallelMergeError(
            "sample-parallel cover must use shard-local flat-token row indices"
        )
    try:
        qname_census = validate_qname_census(
            cover.get("qname_census")  # type: ignore[arg-type]
        )
        execution_identity = validate_execution_identity(
            cover.get("execution_identity"),  # type: ignore[arg-type]
            qname_census=qname_census,
        )
    except SampleParallelProbeError as exc:
        raise SampleParallelMergeError(str(exc)) from exc
    raw_shards = cover.get("shards")
    if (
        not isinstance(raw_shards, Sequence)
        or isinstance(raw_shards, (str, bytes))
        or not raw_shards
    ):
        raise SampleParallelMergeError("sample-parallel cover has no shards")

    global_hash = str(cover.get("global_calibration_hash", ""))
    artifact_hash = str(cover.get("calibration_artifact_sha256", ""))
    partition_count = int(cover.get("partition_count", 0) or 0)
    if not re.fullmatch(r"[0-9a-f]{32}", global_hash):
        raise SampleParallelMergeError(
            "sample-parallel cover has invalid global calibration hash"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
        raise SampleParallelMergeError(
            "sample-parallel cover has invalid calibration artifact hash"
        )
    ordered: list[dict[str, object]] = []
    for shard_index, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping) or set(raw) != _PARTITION_KEYS:
            raise SampleParallelMergeError("sample-parallel shard is malformed")
        try:
            row = _normalize_partition_contract(
                raw, where=f"sample-parallel cover shard {shard_index}"
            )
        except SampleParallelProbeError as exc:
            raise SampleParallelMergeError(str(exc)) from exc
        if row["schema"] != SAMPLE_PARALLEL_SHARD_SCHEMA:
            raise SampleParallelMergeError(
                "unsupported sample-parallel shard schema"
            )
        if (
            row["global_calibration_hash"] != global_hash
            or row["calibration_artifact_sha256"] != artifact_hash
            or row["global_samples"] != total_samples
            or row["seqlen"] != seqlen
            or row["partition_count"] != partition_count
        ):
            raise SampleParallelMergeError(
                "sample-parallel shard global calibration identity differs"
            )
        if not re.fullmatch(
            r"[0-9a-f]{32}", row["local_calibration_hash"]
        ):
            raise SampleParallelMergeError(
                "sample-parallel shard has invalid local calibration hash"
            )
        if not row["dataset"] or not row["model"] or row["calib_seed"] < 0:
            raise SampleParallelMergeError(
                "sample-parallel shard calibration provenance is incomplete"
            )
        ordered.append(row)
    ordered.sort(key=lambda row: int(row["partition_index"]))
    if partition_count != len(ordered) or [
        row["partition_index"] for row in ordered
    ] != list(range(len(ordered))):
        raise SampleParallelMergeError(
            "sample-parallel shard indexes must be exactly 0..N-1"
        )
    cursor = 0
    for row in ordered:
        start = int(row["sample_start"])
        stop = int(row["sample_stop"])
        if start != cursor or stop <= start:
            raise SampleParallelMergeError(
                "sample-parallel shards are not one contiguous disjoint cover: "
                f"expected start {cursor}, observed [{start}, {stop})"
            )
        if row["sample_indices"] != list(range(start, stop)):
            raise SampleParallelMergeError(
                "sample-parallel shard sample_indices differ from its range"
            )
        if row["local_samples"] != stop - start:
            raise SampleParallelMergeError(
                "sample-parallel shard local_samples differs from its range"
            )
        cursor = stop
    if cursor != total_samples:
        raise SampleParallelMergeError(
            "sample-parallel shards do not cover total_samples exactly: "
            f"covered={cursor}, expected={total_samples}"
        )
    for row in ordered:
        if (
            execution_identity["model"] != row["model"]
            or execution_identity["dataset"] != row["dataset"]
            or int(execution_identity["calib_seed"]) != row["calib_seed"]
        ):
            raise SampleParallelMergeError(
                "execution identity model/dataset/seed differs from a "
                "calibration partition"
            )

    normalized: dict[str, object] = {
        "schema": SAMPLE_PARALLEL_COVER_SCHEMA,
        "global_calibration_hash": global_hash,
        "calibration_artifact_sha256": artifact_hash,
        "total_samples": total_samples,
        "seqlen": seqlen,
        "partition_count": partition_count,
        "row_indices_scope": _ROW_INDEX_SCOPE,
        "execution_identity": execution_identity,
        "qname_census": qname_census,
        "shards": ordered,
    }
    supplied_digest = cover.get("identity_sha256")
    expected_digest = canonical_json_sha256(
        normalized, where="sample-parallel probe cover"
    )
    if supplied_digest is None and not allow_missing_identity:
        raise SampleParallelMergeError(
            "sample-parallel cover identity_sha256 is required"
        )
    if supplied_digest is not None and supplied_digest != expected_digest:
        raise SampleParallelMergeError(
            "sample-parallel cover identity_sha256 differs"
        )
    return normalized, tuple(ordered)


def validate_worker_sample_cover(
    cover: Mapping[str, object],
    *,
    run_contract: Mapping[str, object],
    partition_contract: Mapping[str, object],
) -> dict[str, object]:
    """Bind one worker to the precomputed cover before model/GPU setup.

    The selected partition comes from the currently loaded calibration
    artifact, so exact equality also proves that the artifact SHA-256 and
    local/global calibration hashes are the ones signed into the cover.
    """
    normalized_cover, cover_shards = _validate_cover(cover)
    try:
        normalized_run = validate_run_contract(run_contract)
        normalized_partition = _normalize_partition_contract(
            partition_contract, where="sample-parallel worker partition"
        )
    except SampleParallelProbeError as exc:
        raise SampleParallelMergeError(str(exc)) from exc
    if normalized_cover["execution_identity"] != normalized_run[
        "execution_identity"
    ]:
        raise SampleParallelMergeError(
            "sample-parallel worker cover execution identity differs from "
            "the run contract"
        )
    if normalized_cover["qname_census"] != normalized_run["qname_census"]:
        raise SampleParallelMergeError(
            "sample-parallel worker cover qname census differs from the "
            "run contract"
        )
    index = int(normalized_partition["partition_index"])
    if index not in range(len(cover_shards)) or cover_shards[
        index
    ] != normalized_partition:
        raise SampleParallelMergeError(
            "sample-parallel worker partition/calibration artifact differs "
            "from the signed sample cover"
        )
    return {
        **normalized_cover,
        "identity_sha256": str(cover["identity_sha256"]),
    }


def _canonical_payloads(
    payloads: Sequence[Mapping[str, object]],
    cover: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], ...], dict[str, object], str]:
    normalized_cover, cover_shards = _validate_cover(cover)
    cover_digest = canonical_json_sha256(
        normalized_cover, where="sample-parallel probe cover"
    )
    if len(payloads) != len(cover_shards):
        raise SampleParallelMergeError(
            "sample-parallel payload count differs from declared shard cover"
        )
    by_index: dict[int, Mapping[str, object]] = {}
    execution_identity = normalized_cover["execution_identity"]
    seqlen = int(normalized_cover["seqlen"])
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise SampleParallelMergeError("sample-parallel payload is malformed")
        if set(payload) != _PRODUCER_PAYLOAD_KEYS:
            raise SampleParallelMergeError(
                "sample-parallel producer payload fields differ from the "
                "closed dense schema"
            )
        meta = payload.get("meta")
        stats = payload.get("stats")
        if not isinstance(meta, Mapping) or not isinstance(stats, Mapping):
            raise SampleParallelMergeError(
                "sample-parallel payload needs mapping stats and meta"
            )
        if meta.get("h_detail_dir") not in (None, ""):
            raise SampleParallelMergeError(
                "sample-parallel exact merge does not support h-detail shards"
            )
        for key in _ROUTED_PAYLOAD_MAPS:
            value = payload[key]
            if not isinstance(value, Mapping) or value:
                raise SampleParallelMergeError(
                    f"sample-parallel v1 rejects routed/packed payload map {key!r}"
                )
        stamp = meta.get("sample_parallel")
        if not isinstance(stamp, Mapping):
            raise SampleParallelMergeError(
                "sample-parallel payload lacks its exact shard stamp"
            )
        try:
            normalized_stamp = _normalize_partition_contract(
                stamp, where="sample-parallel payload shard stamp"
            )
            index = int(normalized_stamp["partition_index"])
        except SampleParallelProbeError as exc:
            raise SampleParallelMergeError(
                f"sample-parallel shard contract is invalid: {exc}"
            ) from exc
        if index < 0 or index >= len(cover_shards) or index in by_index:
            raise SampleParallelMergeError(
                f"sample-parallel payload shard index {index} is duplicate/out of range"
            )
        expected_shard = cover_shards[index]
        if set(stamp) != set(expected_shard):
            raise SampleParallelMergeError(
                f"sample-parallel payload shard {index} contract fields differ"
            )
        if normalized_stamp != expected_shard:
            raise SampleParallelMergeError(
                f"sample-parallel payload shard {index} differs from the cover"
            )
        if meta.get("sample_parallel_execution_identity") != execution_identity:
            raise SampleParallelMergeError(
                f"sample-parallel payload shard {index} execution identity differs"
            )
        shard_samples = int(expected_shard["sample_stop"]) - int(
            expected_shard["sample_start"]
        )
        if (
            type(meta.get("nsamples")) is not int
            or meta.get("nsamples") != shard_samples
        ):
            raise SampleParallelMergeError(
                f"sample-parallel payload shard {index} nsamples differs"
            )
        if (
            type(meta.get("seqlen")) is not int
            or meta.get("seqlen") != seqlen
        ):
            raise SampleParallelMergeError(
                f"sample-parallel payload shard {index} seqlen differs"
            )
        if meta.get("calib_hash") != expected_shard["local_calibration_hash"]:
            raise SampleParallelMergeError(
                f"sample-parallel payload shard {index} calib_hash differs"
            )
        if (
            type(meta.get("fisher_norm_tokens")) is not int
            or meta.get("fisher_norm_tokens") != shard_samples * seqlen
        ):
            raise SampleParallelMergeError(
                f"sample-parallel payload shard {index} Fisher denominator differs"
            )
        required_meta = {
            "model": execution_identity["model"],
            "dataset": execution_identity["dataset"],
            "dtype": execution_identity["dtype"],
            "importance_weighting": execution_identity["importance_weighting"],
            "activation_rows_limit": execution_identity["activation_rows_limit"],
            "emit_marginals": execution_identity["emit_marginals"],
            "calibration_modality": execution_identity["calibration_modality"],
            "packed_fisher_estimator": "per_token_v2",
            "sample_parallel_activation_scope": execution_identity[
                "activation_scope"
            ],
        }
        for key, expected in required_meta.items():
            if meta.get(key) != expected:
                raise SampleParallelMergeError(
                    f"sample-parallel payload shard {index} execution meta "
                    f"{key!r} differs"
                )
        by_index[index] = payload
    return (
        tuple(by_index[index] for index in range(len(cover_shards))),
        normalized_cover,
        cover_digest,
    )


def _array_shape(value: object) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _nonnegative_finite_array(value: object, *, where: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise SampleParallelMergeError(f"{where} is not an array") from exc
    if (
        array.dtype.kind not in "fiu"
        or not np.all(np.isfinite(array))
        or np.any(array < 0)
    ):
        raise SampleParallelMergeError(
            f"{where} must contain only finite nonnegative raw statistics"
        )
    return array


def _finite_nonnegative_scalar(value: object, *, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise SampleParallelMergeError(
            f"{where} must be a numeric scalar"
        )
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise SampleParallelMergeError(
            f"{where} must be finite and nonnegative"
        )
    return result


def _validate_fisher_marginal_trace(
    row: Mapping[str, object],
    *,
    where: str,
) -> None:
    """Require both Fisher marginals to reduce to their raw scalar trace."""
    trace = float(row["h_trace_raw"])
    for field in ("fisher_row", "fisher_col"):
        total = float(np.asarray(row[field], dtype=np.float64).sum())
        if not math.isclose(
            total,
            trace,
            rel_tol=_FISHER_MARGINAL_REL_TOL,
            abs_tol=0.0,
        ):
            raise SampleParallelMergeError(
                f"{where} sum({field})={total:.9g} differs from "
                f"h_trace_raw={trace:.9g} beyond the qualified fp32 "
                f"relative tolerance {_FISHER_MARGINAL_REL_TOL:g}"
            )


def _merge_stat_rows(
    rows: Sequence[Mapping[str, object]],
    global_tokens: int,
    *,
    expected_n_tokens: Sequence[int],
    qname: str,
    require_marginals: bool,
    expected_shape: tuple[int, int],
) -> dict:
    first = rows[0]
    if len(rows) != len(expected_n_tokens):
        raise AssertionError("stat/token-cover cardinality differs")
    expected_fields = set(_DENSE_STAT_BASE_FIELDS)
    if require_marginals:
        expected_fields.update(_DENSE_MARGINAL_FIELDS)
    expected_out, expected_in = (int(expected_shape[0]), int(expected_shape[1]))
    for field in _SCALAR_SUM_FIELDS:
        if any(field not in row for row in rows):
            raise SampleParallelMergeError(
                f"sample-parallel stat row lacks required raw field {field!r}"
            )
    for index, (row, expected_tokens) in enumerate(zip(
        rows, expected_n_tokens, strict=True
    )):
        if set(row) != expected_fields:
            raise SampleParallelMergeError(
                f"sample-parallel {qname} shard {index} stat fields differ "
                "from the closed dense schema"
            )
        for field in ("h_trace_raw", "h_w2_sum_raw"):
            try:
                value = _finite_nonnegative_scalar(
                    row[field],
                    where=(
                        f"sample-parallel {qname} shard {index} raw {field}"
                    ),
                )
            except KeyError as exc:
                raise SampleParallelMergeError(
                    f"sample-parallel {qname} shard {index} raw {field} is invalid"
                ) from exc
        for field in ("w_max_abs", "w_norm_sq"):
            try:
                value = _finite_nonnegative_scalar(
                    row[field],
                    where=f"sample-parallel {qname} shard {index} {field}",
                )
            except KeyError as exc:
                raise SampleParallelMergeError(
                    f"sample-parallel {qname} shard {index} {field} is invalid"
                ) from exc
        for field in ("n_params", "in_features", "out_features"):
            value = row.get(field)
            if type(value) is not int or value < 1:
                raise SampleParallelMergeError(
                    f"sample-parallel {qname} shard {index} {field} is invalid"
                )
        if (
            row["out_features"] != expected_out
            or row["in_features"] != expected_in
            or row["n_params"] != expected_out * expected_in
        ):
            raise SampleParallelMergeError(
                f"sample-parallel {qname} shard {index} geometry differs "
                "from the source qname manifest"
            )
        seen = row.get("n_tokens_seen")
        if type(seen) is not int or seen != int(expected_tokens):
            raise SampleParallelMergeError(
                f"sample-parallel {qname} shard {index} n_tokens_seen differs: "
                f"observed={seen!r}, expected={expected_tokens}"
            )
        forbidden = sorted(set(row) & _PACKED_MARGINAL_FIELDS)
        if forbidden:
            raise SampleParallelMergeError(
                f"sample-parallel v1 rejects routed/packed stats for {qname}: "
                f"{forbidden[:8]}"
            )
        routed = sorted(
            marker for marker in _ROUTED_NULLABLE_MARKERS
            if row[marker] is not None
        )
        routed.extend(sorted(set(row) & _PACKED_PRESENCE_MARKERS))
        if routed:
            raise SampleParallelMergeError(
                f"sample-parallel v1 rejects routed/packed markers for {qname}: "
                f"{routed[:8]}"
            )
        present_marginals = set(row) & _DENSE_MARGINAL_FIELDS
        if require_marginals and present_marginals != _DENSE_MARGINAL_FIELDS:
            raise SampleParallelMergeError(
                f"sample-parallel {qname} shard {index} dense marginals are incomplete"
            )
        if not require_marginals and present_marginals:
            raise SampleParallelMergeError(
                f"sample-parallel {qname} shard {index} emitted undeclared marginals"
            )
        for field in present_marginals:
            array = _nonnegative_finite_array(
                row[field], where=f"sample-parallel {qname} shard {index} {field}"
            )
            expected_length = (
                expected_out
                if field in {"fisher_row", "g_sq_sum"}
                else expected_in
            )
            if array.dtype != np.dtype(np.float32) or array.shape != (
                expected_length,
            ):
                raise SampleParallelMergeError(
                    f"sample-parallel {qname} shard {index} {field} "
                    "dtype/shape differs from the source qname manifest"
                )
        if require_marginals:
            _validate_fisher_marginal_trace(
                row,
                where=f"sample-parallel {qname} shard {index}",
            )
    excluded = (
        set(_SCALAR_SUM_FIELDS)
        | _DERIVED_FIELDS
        | _DENSE_MARGINAL_FIELDS
        | _PACKED_MARGINAL_FIELDS
    )
    static = {key: first[key] for key in first if key not in excluded}
    for row in rows[1:]:
        observed = {key: row[key] for key in row if key not in excluded}
        if observed != static:
            raise SampleParallelMergeError(
                "sample-parallel stat static metadata differs"
            )

    merged = copy.deepcopy(dict(first))
    for field in _DERIVED_FIELDS:
        merged.pop(field, None)
    merged["h_trace_raw"] = sum(float(row["h_trace_raw"]) for row in rows)
    merged["h_w2_sum_raw"] = sum(float(row["h_w2_sum_raw"]) for row in rows)
    merged["n_tokens_seen"] = sum(int(row["n_tokens_seen"]) for row in rows)
    if not np.isfinite(merged["h_trace_raw"]) or not np.isfinite(
        merged["h_w2_sum_raw"]
    ):
        raise SampleParallelMergeError(
            f"sample-parallel merged raw scalar overflow for {qname}"
        )

    for field in _DENSE_MARGINAL_FIELDS:
        present = [field in row for row in rows]
        if any(present) and not all(present):
            raise SampleParallelMergeError(
                f"sample-parallel dense marginal {field!r} presence differs"
            )
        if all(present):
            shapes = {_array_shape(row[field]) for row in rows}
            if len(shapes) != 1:
                raise SampleParallelMergeError(
                    f"sample-parallel dense marginal {field!r} shape differs"
                )
    for field in _DENSE_MARGINAL_FIELDS:
        merged.pop(field, None)
    for row in rows:
        merge_marginals(merged, row)
    for field in _DENSE_MARGINAL_FIELDS & set(merged):
        _nonnegative_finite_array(
            merged[field], where=f"sample-parallel merged {qname} {field}"
        )
    if require_marginals:
        _validate_fisher_marginal_trace(
            merged, where=f"sample-parallel merged {qname}"
        )

    finalize_fisher_stats({"row": merged}, global_tokens)
    for field in ("h_trace", "h_w2_sum"):
        value = float(merged.get(field, float("nan")))
        if not np.isfinite(value) or value < 0.0:
            raise SampleParallelMergeError(
                f"sample-parallel finalized {qname} {field} is invalid"
            )
    return merged


def validate_sample_parallel_stat_row(
    row: Mapping[str, object],
    *,
    qname: str,
    expected_tokens: int,
    expected_shape: tuple[int, int],
    require_marginals: bool,
) -> None:
    """Validate one worker row with the reducer's closed dense contract."""
    _merge_stat_rows(
        [row],
        int(expected_tokens),
        expected_n_tokens=[int(expected_tokens)],
        qname=str(qname),
        require_marginals=bool(require_marginals),
        expected_shape=tuple(int(value) for value in expected_shape),
    )


def validate_sample_parallel_merged_stat_row(
    row: Mapping[str, object],
    *,
    qname: str,
    expected_tokens: int,
    normalization_tokens: int,
    expected_shape: tuple[int, int],
    require_marginals: bool,
) -> None:
    """Validate a globally reduced row, including its derived normalization."""
    expected_fields = set(_DENSE_STAT_BASE_FIELDS) - {"route_prob"}
    if require_marginals:
        expected_fields.update(_DENSE_MARGINAL_FIELDS)
    if not isinstance(row, Mapping) or set(row) != expected_fields:
        raise SampleParallelMergeError(
            f"sample-parallel merged {qname} stat fields differ"
        )
    worker_form = {**dict(row), "route_prob": None}
    rebuilt = _merge_stat_rows(
        [worker_form],
        int(normalization_tokens),
        expected_n_tokens=[int(expected_tokens)],
        qname=str(qname),
        require_marginals=bool(require_marginals),
        expected_shape=tuple(int(value) for value in expected_shape),
    )
    for field in ("h_trace", "h_w2_sum"):
        if (
            isinstance(row[field], (bool, np.bool_))
            or not isinstance(
                row[field], (int, float, np.integer, np.floating)
            )
            or float(row[field]) != float(rebuilt[field])
        ):
            raise SampleParallelMergeError(
                f"sample-parallel merged {qname} {field} normalization differs"
            )
    if (
        type(row["h_trace_norm_tokens"]) is not int
        or row["h_trace_norm_tokens"] != int(normalization_tokens)
    ):
        raise SampleParallelMergeError(
            f"sample-parallel merged {qname} Fisher normalization differs"
        )


def merge_sample_parallel_probe_payloads(
    payloads: Sequence[Mapping[str, object]],
    *,
    expected_cover: Mapping[str, object],
) -> dict[str, object]:
    """Merge same-qname probe payloads over one exact disjoint sample cover."""
    ordered, cover, cover_digest = _canonical_payloads(payloads, expected_cover)
    probe_manifest = cover["qname_census"]["probe_qname_manifest"]
    terminal_manifest = cover["qname_census"]["terminal_qname_manifest"]
    activation_manifest = cover["qname_census"]["activation_qname_manifest"]
    expected_entries = probe_manifest["entries"]
    expected = set(expected_entries)
    for index, payload in enumerate(ordered):
        observed = set(str(name) for name in payload["stats"])
        if observed != expected:
            raise SampleParallelMergeError(
                f"sample-parallel payload shard {index} qname cover differs: "
                f"missing={sorted(expected - observed)[:8]} "
                f"unexpected={sorted(observed - expected)[:8]}"
            )

    importance_weighting = bool(
        cover["execution_identity"]["importance_weighting"]
    )
    importance_execution_sha = importance_execution_identity_sha256(
        cover["execution_identity"], cover["qname_census"]
    )
    importance_receipt: dict[str, object] | None = None
    if importance_weighting:
        for index, payload in enumerate(ordered):
            raw = payload["meta"].get("sample_parallel_importance")
            if not isinstance(raw, Mapping):
                raise SampleParallelMergeError(
                    "importance-weighted sample shards require a global "
                    "importance-normalization receipt"
                )
            try:
                receipt = validate_global_importance_receipt(
                    raw,
                    partition_contract=cover["shards"][index],
                    execution_identity_sha256=importance_execution_sha,
                )
            except SampleParallelProbeError as exc:
                raise SampleParallelMergeError(
                    f"sample-parallel payload shard {index} has an invalid "
                    f"importance-normalization receipt: {exc}"
                ) from exc
            if importance_receipt is None:
                importance_receipt = receipt
            elif receipt != importance_receipt:
                raise SampleParallelMergeError(
                    "sample shards used different importance-normalization receipts"
                )
    else:
        for index, payload in enumerate(ordered):
            if payload["meta"].get("sample_parallel_importance") is not None:
                raise SampleParallelMergeError(
                    f"unweighted sample shard {index} carries an importance receipt"
                )

    global_tokens = int(cover["total_samples"]) * int(cover["seqlen"])
    seqlen = int(cover["seqlen"])
    shard_samples = [int(shard["local_samples"]) for shard in cover["shards"]]
    require_marginals = bool(cover["execution_identity"]["emit_marginals"])
    merged_stats: dict[str, dict] = {}
    for qname in sorted(expected):
        entry = expected_entries[qname]
        disposition = entry["disposition"]
        if disposition not in {
            BODY_STATS_AND_ACTIVATION, LM_HEAD_STATS_ONLY, MTP_STATS_ONLY,
        }:
            raise SampleParallelMergeError(
                f"probe manifest admits unsupported disposition for {qname}"
            )
        per_sample = token_rows_per_sample(
            str(entry["token_rows_per_sample"]), seqlen
        )
        merged_stats[qname] = _merge_stat_rows(
            [payload["stats"][qname] for payload in ordered],
            global_tokens,
            expected_n_tokens=[count * per_sample for count in shard_samples],
            qname=qname,
            require_marginals=require_marginals,
            expected_shape=tuple(int(dim) for dim in entry["shape"]),
        )

    first_meta = copy.deepcopy(dict(ordered[0]["meta"]))
    for key in (
        "sample_parallel", "calib_hash", "calib_hashes", "dataset",
        "activation_cache_dir", "h_detail_dir", "shards", "n_shards",
    ):
        first_meta.pop(key, None)
    first_meta.update({
        "nsamples": int(cover["total_samples"]),
        "seqlen": int(cover["seqlen"]),
        "dataset": cover["execution_identity"]["dataset"],
        "calib_seed": int(cover["execution_identity"]["calib_seed"]),
        "fisher_norm_tokens": global_tokens,
        "calib_hash": str(cover["global_calibration_hash"]),
        "calib_hashes": [str(cover["global_calibration_hash"])],
        "sample_parallel_merge": {
            "schema": SAMPLE_PARALLEL_MERGE_SCHEMA,
            "cover": copy.deepcopy(cover),
            "cover_identity_sha256": cover_digest,
            "exact_disjoint_cover": True,
            "canonical_shard_order": list(range(len(ordered))),
            "overlapping_qname_stats_merged": True,
            "global_fisher_normalization_applied_once": True,
            "importance_normalization_receipt": importance_receipt,
            "probe_qname_manifest_sha256": probe_manifest["identity_sha256"],
            "activation_qname_manifest_sha256": activation_manifest[
                "identity_sha256"
            ],
            "terminal_qname_manifest_sha256": terminal_manifest[
                "identity_sha256"
            ],
            "terminal_bf16_lm_head_and_mtp_complete": True,
            "routed_packed_stats_rejected": True,
        },
    })
    return {
        "stats": merged_stats,
        "router_counts": {},
        "router_totals": {},
        "router_active_counts": {},
        "expert_route_stats": {},
        "expert_info": {},
        "meta": first_meta,
    }


def _tensor_identity(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().to("cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": [int(dim) for dim in value.shape],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _load_activation_dir(path: Path) -> dict[str, dict[str, object]]:
    if path.is_symlink() or not path.is_dir():
        raise SampleParallelMergeError(f"activation cache is not a directory: {path}")
    entries: dict[str, dict[str, object]] = {}
    files = sorted(path.iterdir())
    unknown = [
        item for item in files
        if item.is_symlink() or not item.is_file() or item.suffix != ".pt"
    ]
    if unknown:
        raise SampleParallelMergeError(
            "activation cache contains an unknown or unsafe entry: "
            f"{unknown[0]}"
        )
    for file_path in files:
        blob = torch.load(file_path, map_location="cpu", weights_only=False)
        if not isinstance(blob, Mapping) or set(blob) != _ACTIVATION_BLOB_KEYS:
            raise SampleParallelMergeError(f"activation blob is malformed: {file_path}")
        name = str(blob.get("name", ""))
        inputs = blob.get("inputs")
        if not name or not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise SampleParallelMergeError(f"activation blob is malformed: {file_path}")
        if not torch.is_floating_point(inputs) or not bool(
            torch.isfinite(inputs).all().item()
        ):
            raise SampleParallelMergeError(
                f"activation inputs must be finite for {name}"
            )
        expected_name = _ACTIVATION_FNAME_SUB.sub("__", name) + ".pt"
        if file_path.name != expected_name or name in entries:
            raise SampleParallelMergeError(
                f"activation blob filename/name collision at {file_path}"
            )
        row_indices = blob.get("row_indices")
        if (
            not isinstance(row_indices, torch.Tensor)
            or row_indices.ndim != 1
            or row_indices.dtype != torch.int64
            or int(row_indices.numel()) != int(inputs.shape[0])
        ):
            raise SampleParallelMergeError(
                f"activation row_indices are malformed for {name}"
            )
        row_priorities = blob.get("row_priorities")
        if (
            not isinstance(row_priorities, torch.Tensor)
            or row_priorities.ndim != 1
            or row_priorities.dtype != torch.int64
            or int(row_priorities.numel()) != int(inputs.shape[0])
        ):
            raise SampleParallelMergeError(
                f"activation row_priorities are malformed for {name}"
            )
        stamp = blob.get("sample_parallel_activation")
        if not isinstance(stamp, Mapping) or set(stamp) != _ACTIVATION_STAMP_KEYS:
            raise SampleParallelMergeError(
                f"activation priority stamp is malformed for {name}"
            )
        entries[name] = {
            "inputs": inputs.detach().to("cpu").contiguous(),
            "row_indices": row_indices.detach().to("cpu").contiguous(),
            "row_priorities": row_priorities.detach().to("cpu").contiguous(),
            "sample_parallel_activation": copy.deepcopy(dict(stamp)),
        }
    return entries


def _publish_directory_no_clobber(temporary: Path, destination: Path) -> None:
    """Atomically publish a completed directory with Linux RENAME_NOREPLACE."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        # PrismaQuant's qualified GPU environments are Linux/glibc.  A
        # platform without renameat2 cannot provide the publication invariant,
        # so fail rather than fall back to an existence-check race.
        raise SampleParallelMergeError(
            "atomic no-clobber directory publication requires renameat2"
        )
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(temporary), -100, os.fsencode(destination), 1
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise SampleParallelMergeError(
                f"activation-cache merge output already exists: {destination}"
            )
        raise OSError(error, os.strerror(error), str(destination))
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def merge_sample_parallel_activation_caches(
    cache_dirs_by_shard: Mapping[int, str | Path],
    output_dir: str | Path,
    *,
    expected_cover: Mapping[str, object],
    max_rows: int | None = None,
) -> dict[str, object]:
    """Union ordinary ``{inputs,name,row_indices}`` activation-cache blobs."""
    cover, cover_shards = _validate_cover(expected_cover)
    cover_digest = canonical_json_sha256(
        cover, where="sample-parallel activation-cache cover"
    )
    indexes = set(int(index) for index in cache_dirs_by_shard)
    expected_indexes = set(range(len(cover_shards)))
    if indexes != expected_indexes:
        raise SampleParallelMergeError(
            "activation-cache shard directories differ from the exact cover"
        )
    activation_manifest = cover["qname_census"]["activation_qname_manifest"]
    expected = set(activation_manifest["entries"])
    cache_dtype_contract = cover["execution_identity"]["activation_scope"].get(
        "cache_dtype"
    )
    if cache_dtype_contract != "torch.float32":
        raise SampleParallelMergeError(
            "activation cache dtype contract is unsupported"
        )
    expected_cache_dtype = torch.float32
    if max_rows is None or int(max_rows) < 1:
        raise SampleParallelMergeError(
            "exact activation merge requires a positive global max_rows"
        )
    declared_rows = int(cover["execution_identity"]["activation_rows_limit"])
    if int(max_rows) != declared_rows:
        raise SampleParallelMergeError(
            "activation merge max_rows differs from the closed execution "
            f"identity: observed={max_rows}, expected={declared_rows}"
        )

    loaded = {
        index: _load_activation_dir(Path(cache_dirs_by_shard[index]))
        for index in sorted(expected_indexes)
    }
    observed_union = set().union(*(set(entries) for entries in loaded.values()))
    if observed_union != expected:
        raise SampleParallelMergeError(
            "activation-cache qname union differs from expected cover: "
            f"missing={sorted(expected - observed_union)[:8]} "
            f"unexpected={sorted(observed_union - expected)[:8]}"
        )
    for index, entries in loaded.items():
        observed = set(entries)
        if observed != expected:
            raise SampleParallelMergeError(
                f"dense activation-cache shard {index} qname cover differs: "
                f"missing={sorted(expected - observed)[:8]} "
                f"unexpected={sorted(observed - expected)[:8]}"
            )

    output = Path(output_dir)
    if output.exists():
        raise SampleParallelMergeError(
            f"activation-cache merge output already exists: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.sample-merge-", dir=output.parent
    ))
    records: dict[str, object] = {}
    retained_rows_by_priority_group: dict[str, torch.Tensor] = {}
    expected_local_plan_cache: dict[
        tuple[int, str, int, int], tuple[np.ndarray, np.ndarray]
    ] = {}
    try:
        for qname in sorted(expected):
            source_shape = activation_manifest["entries"][qname]["shape"]
            expected_width = int(source_shape[1])
            parts: list[torch.Tensor] = []
            indices: list[torch.Tensor] = []
            priorities: list[torch.Tensor] = []
            dtype = None
            width = None
            contributing: list[int] = []
            for shard in cover_shards:
                index = int(shard["partition_index"])
                entry = loaded[index].get(qname)
                if entry is None:  # guarded by the exact per-shard cover above
                    raise AssertionError("validated activation qname disappeared")
                value = entry["inputs"]
                if (
                    value.dtype != expected_cache_dtype
                    or int(value.shape[1]) != expected_width
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise SampleParallelMergeError(
                        f"activation-cache dtype/width for {qname} differs "
                        "from the closed source/cache contract"
                    )
                if dtype is None:
                    dtype, width = value.dtype, int(value.shape[1])
                elif value.dtype != dtype or int(value.shape[1]) != width:
                    raise SampleParallelMergeError(
                        f"activation-cache dtype/width differs for {qname}"
                    )
                parts.append(value)
                contributing.append(index)
                local_indices = entry.get("row_indices")
                local_priorities = entry.get("row_priorities")
                stamp = entry.get("sample_parallel_activation")
                if (
                    not isinstance(local_indices, torch.Tensor)
                    or not isinstance(local_priorities, torch.Tensor)
                    or not isinstance(stamp, Mapping)
                ):
                    raise SampleParallelMergeError(
                        f"activation-cache shard {index} for {qname} lacks "
                        "exact global-priority sampling provenance"
                    )
                if (
                    stamp.get("schema") != ACTIVATION_CACHE_SHARD_SCHEMA
                    or stamp.get("priority_schema") != ACTIVATION_PRIORITY_SCHEMA
                    or stamp.get("selection") != "local_top_r"
                    or stamp.get("row_indices_scope") != _ROW_INDEX_SCOPE
                    or stamp.get("layout") != "dense_linear"
                    or stamp.get("priority_group")
                    != activation_priority_group(qname)
                    or stamp.get("global_calibration_hash")
                    != cover["global_calibration_hash"]
                    or stamp.get("execution_identity_sha256")
                    != cover["execution_identity"]["identity_sha256"]
                    or type(stamp.get("partition_index")) is not int
                    or stamp.get("partition_index") != index
                    or type(stamp.get("rows_limit")) is not int
                    or type(stamp.get("candidate_rows")) is not int
                ):
                    raise SampleParallelMergeError(
                        f"activation-cache shard {index} for {qname} has "
                        "invalid priority-sampling provenance"
                    )
                local_limit = stamp["rows_limit"]
                candidate_rows = stamp["candidate_rows"]
                if local_limit < 1 or candidate_rows < 0 or int(
                    value.shape[0]
                ) != min(local_limit, candidate_rows):
                    raise SampleParallelMergeError(
                        f"activation-cache shard {index} for {qname} has "
                        "an invalid local top-R cardinality"
                    )
                if local_limit != declared_rows:
                    raise SampleParallelMergeError(
                        f"activation-cache shard {index} local rows limit "
                        "differs from the execution identity"
                    )
                shard_tokens = (
                    int(shard["sample_stop"]) - int(shard["sample_start"])
                ) * int(cover["seqlen"])
                if local_indices.numel() and (
                    int(local_indices.min().item()) < 0
                    or int(local_indices.max().item()) >= shard_tokens
                ):
                    raise SampleParallelMergeError(
                        f"activation-cache row index is outside shard {index} "
                        f"for {qname}"
                    )
                if candidate_rows != shard_tokens:
                    raise SampleParallelMergeError(
                        f"activation-cache candidate rows differ from the "
                        f"complete dense sample/token grid for shard {index}, "
                        f"{qname}"
                    )
                offset = int(shard["sample_start"]) * int(cover["seqlen"])
                global_indices = local_indices + offset
                global_indices_np = global_indices.numpy()
                observed_priorities_np = local_priorities.numpy()
                expected_priorities_np = (
                    _reference_activation_priorities_numpy(
                        str(cover["global_calibration_hash"]),
                        qname,
                        global_indices_np,
                    )
                )
                if not np.array_equal(
                    observed_priorities_np, expected_priorities_np
                ):
                    raise SampleParallelMergeError(
                        f"activation-cache shard {index} priorities differ "
                        f"from global row identities for {qname}"
                    )
                priority_group = activation_priority_group(qname)
                plan_key = (
                    index, priority_group, local_limit, candidate_rows,
                )
                expected_plan = expected_local_plan_cache.get(plan_key)
                if expected_plan is None:
                    all_global_np = (
                        np.arange(candidate_rows, dtype=np.int64)
                        + np.int64(offset)
                    )
                    expected_global_np, expected_plan_priorities_np = (
                        _reference_top_r_numpy(
                            str(cover["global_calibration_hash"]),
                            qname,
                            all_global_np,
                            local_limit,
                        )
                    )
                    expected_plan = (
                        expected_global_np - np.int64(offset),
                        expected_plan_priorities_np,
                    )
                    expected_local_plan_cache[plan_key] = expected_plan
                expected_local_np, expected_plan_priorities_np = expected_plan
                if not np.array_equal(
                    local_indices.numpy(), expected_local_np
                ):
                    raise SampleParallelMergeError(
                        f"activation-cache shard {index} rows are not the "
                        f"declared dense local top-R for {qname}"
                    )
                if not np.array_equal(
                    observed_priorities_np, expected_plan_priorities_np
                ):
                    raise SampleParallelMergeError(
                        f"activation-cache shard {index} rows are not in "
                        f"canonical local priority order for {qname}"
                    )
                indices.append(global_indices)
                priorities.append(local_priorities)
            if not parts:
                raise SampleParallelMergeError(
                    f"activation-cache expected qname {qname!r} has no rows"
                )
            inputs = torch.cat(parts, dim=0).contiguous()
            if not int(inputs.shape[0]):
                raise SampleParallelMergeError(
                    f"activation-cache expected qname {qname!r} has no rows"
                )
            merged_indices = torch.cat(indices, dim=0).contiguous()
            merged_priorities = torch.cat(priorities, dim=0).contiguous()
            if int(torch.unique(merged_indices).numel()) != int(
                merged_indices.numel()
            ):
                raise SampleParallelMergeError(
                    f"activation-cache global row indices overlap for {qname}"
                )
            union_rows_np = merged_indices.numpy()
            union_priorities_np = (
                _reference_activation_priorities_numpy(
                    str(cover["global_calibration_hash"]),
                    qname,
                    union_rows_np,
                )
            )
            if not np.array_equal(
                merged_priorities.numpy(), union_priorities_np
            ):
                raise SampleParallelMergeError(
                    f"activation-cache union priorities differ for {qname}"
                )
            # Exact top-K is associative over a disjoint union when each
            # partition retains at least K: any row discarded locally has K
            # strictly smaller rows in the same partition.  The PRP makes all
            # priorities unique, so no tie/stable-sort convention is hidden in
            # that proof.  Every local operand was independently established
            # above; this is therefore the exact global top-R union.
            keep = min(int(max_rows), int(union_priorities_np.size))
            order_np = np.argsort(
                union_priorities_np, kind="stable"
            )[:keep].copy()
            order = torch.from_numpy(order_np).to(dtype=torch.long)
            inputs = inputs.index_select(0, order).contiguous()
            merged_indices = merged_indices.index_select(0, order).contiguous()
            merged_priorities = merged_priorities.index_select(
                0, order
            ).contiguous()
            priority_group = activation_priority_group(qname)
            prior_group_rows = retained_rows_by_priority_group.get(priority_group)
            if prior_group_rows is None:
                retained_rows_by_priority_group[priority_group] = merged_indices.clone()
            elif not torch.equal(prior_group_rows, merged_indices):
                raise SampleParallelMergeError(
                    f"activation-cache fused siblings in {priority_group!r} "
                    "did not retain identical global rows"
                )
            payload: dict[str, object] = {
                "inputs": inputs,
                "name": qname,
                "row_indices": merged_indices,
                "row_priorities": merged_priorities,
            }
            identity = {
                "inputs": _tensor_identity(inputs),
                "row_indices": (
                    _tensor_identity(merged_indices)
                ),
                "row_priorities": _tensor_identity(merged_priorities),
                "source_shards": contributing,
                "priority_group": activation_priority_group(qname),
            }
            identity["identity_sha256"] = canonical_json_sha256(
                identity, where=f"sample-parallel activation {qname}"
            )
            payload["sample_parallel_merge"] = {
                "schema": ACTIVATION_CACHE_MERGE_SCHEMA,
                "cover_identity_sha256": cover_digest,
                "identity_sha256": identity["identity_sha256"],
                "priority_schema": ACTIVATION_PRIORITY_SCHEMA,
                "priority_group": activation_priority_group(qname),
            }
            target = temp / (_ACTIVATION_FNAME_SUB.sub("__", qname) + ".pt")
            torch.save(payload, target)
            records[qname] = identity

        manifest: dict[str, object] = {
            "schema": ACTIVATION_CACHE_MERGE_SCHEMA,
            "cover": cover,
            "cover_identity_sha256": cover_digest,
            "exact_disjoint_cover": True,
            "exact_global_priority_top_r": True,
            "local_global_top_r_associativity_verified": True,
            "independent_numpy_priority_verifier": True,
            "priority_schema": ACTIVATION_PRIORITY_SCHEMA,
            "supported_activation_layout": "dense_linear",
            "routed_packed_activation_caches_rejected": True,
            "fused_group_alignment_verified": True,
            "qnames": len(records),
            "max_rows": int(max_rows) if max_rows is not None else None,
            "records": records,
            "activation_qname_manifest_sha256": activation_manifest[
                "identity_sha256"
            ],
            "source_census_sha256": cover["qname_census"][
                "source_census"
            ]["identity_sha256"],
        }
        manifest["identity_sha256"] = canonical_json_sha256(
            manifest, where="sample-parallel activation-cache manifest"
        )
        (temp / ACTIVATION_CACHE_MANIFEST).write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        for published_file in temp.iterdir():
            if published_file.is_file():
                with published_file.open("rb") as handle:
                    os.fsync(handle.fileno())
        temp_descriptor = os.open(temp, os.O_RDONLY)
        try:
            os.fsync(temp_descriptor)
        finally:
            os.close(temp_descriptor)
        _publish_directory_no_clobber(temp, output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return manifest


def validate_merged_activation_cache_output(
    output_dir: str | Path,
    *,
    expected_cover: Mapping[str, object],
) -> dict[str, object]:
    """Validate an already-published merge directory for owned retry reuse."""
    cover, cover_shards = _validate_cover(expected_cover)
    cover_digest = canonical_json_sha256(
        cover, where="sample-parallel activation-cache cover"
    )
    output = Path(output_dir)
    if output.is_symlink() or not output.is_dir():
        raise SampleParallelMergeError(
            "published activation-cache merge is not a regular directory"
        )
    manifest_path = output / ACTIVATION_CACHE_MANIFEST
    try:
        manifest = _strict_json_loads(
            manifest_path.read_text(encoding="utf-8"),
            where="published activation-cache merge manifest",
        )
    except SampleParallelProbeError as exc:
        raise SampleParallelMergeError(str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise SampleParallelMergeError(
            "published activation-cache merge manifest is unreadable"
        ) from exc
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != _ACTIVATION_MERGE_MANIFEST_KEYS
    ):
        raise SampleParallelMergeError(
            "published activation-cache merge manifest fields differ"
        )
    body = dict(manifest)
    identity_digest = body.pop("identity_sha256", None)
    activation_manifest = cover["qname_census"][
        "activation_qname_manifest"
    ]
    declared_max_rows = cover["execution_identity"][
        "activation_rows_limit"
    ]
    total_candidate_rows = int(cover["total_samples"]) * int(
        cover["seqlen"]
    )
    if (
        type(declared_max_rows) is not int
        or declared_max_rows < 1
        or total_candidate_rows < 1
        or total_candidate_rows > _REFERENCE_U32_DOMAIN
        or identity_digest != canonical_json_sha256(
            body, where="published sample-parallel activation-cache manifest"
        )
        or manifest.get("schema") != ACTIVATION_CACHE_MERGE_SCHEMA
        or manifest.get("cover_identity_sha256") != cover_digest
        or manifest.get("cover") != cover
        or manifest.get("priority_schema") != ACTIVATION_PRIORITY_SCHEMA
        or manifest.get("exact_disjoint_cover") is not True
        or manifest.get("exact_global_priority_top_r") is not True
        or manifest.get("independent_numpy_priority_verifier") is not True
        or manifest.get("local_global_top_r_associativity_verified") is not True
        or manifest.get("supported_activation_layout") != "dense_linear"
        or manifest.get("routed_packed_activation_caches_rejected") is not True
        or manifest.get("fused_group_alignment_verified") is not True
        or type(manifest.get("qnames")) is not int
        or manifest.get("qnames") != len(activation_manifest["entries"])
        or type(manifest.get("max_rows")) is not int
        or manifest.get("max_rows") != declared_max_rows
        or manifest.get("activation_qname_manifest_sha256")
        != activation_manifest["identity_sha256"]
        or manifest.get("source_census_sha256")
        != cover["qname_census"]["source_census"]["identity_sha256"]
    ):
        raise SampleParallelMergeError(
            "published activation-cache merge identity differs"
        )
    records = manifest.get("records")
    expected_qnames = set(
        activation_manifest["entries"]
    )
    if not isinstance(records, Mapping) or set(records) != expected_qnames:
        raise SampleParallelMergeError(
            "published activation-cache merge qname records differ"
        )
    expected_files = {
        _ACTIVATION_FNAME_SUB.sub("__", name) + ".pt"
        for name in expected_qnames
    } | {ACTIVATION_CACHE_MANIFEST}
    output_entries = list(output.iterdir())
    observed_files = {path.name for path in output_entries}
    if observed_files != expected_files or any(
        path.is_symlink() or not path.is_file() for path in output_entries
    ):
        raise SampleParallelMergeError(
            "published activation-cache merge file cover differs"
        )
    contributing = [int(row["partition_index"]) for row in cover_shards]
    expected_cardinality = min(declared_max_rows, total_candidate_rows)
    expected_top_r_by_group: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    observed_rows_by_group: dict[str, torch.Tensor] = {}
    replayed_records: dict[str, object] = {}
    for qname in sorted(expected_qnames):
        path = output / (_ACTIVATION_FNAME_SUB.sub("__", qname) + ".pt")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise SampleParallelMergeError(
                f"published activation-cache payload is unreadable for {qname}"
            ) from exc
        if not isinstance(payload, Mapping) or set(payload) != {
            "inputs", "name", "row_indices", "row_priorities",
            "sample_parallel_merge",
        }:
            raise SampleParallelMergeError(
                f"published activation-cache payload fields differ for {qname}"
            )
        inputs = payload["inputs"]
        rows = payload["row_indices"]
        priorities = payload["row_priorities"]
        if not all(isinstance(value, torch.Tensor) for value in (
            inputs, rows, priorities
        )):
            raise SampleParallelMergeError(
                f"published activation-cache tensors differ for {qname}"
            )
        expected_width = int(
            cover["qname_census"]["activation_qname_manifest"]
            ["entries"][qname]["shape"][1]
        )
        if (
            inputs.dtype != torch.float32
            or inputs.ndim != 2
            or int(inputs.shape[1]) != expected_width
            or not bool(torch.isfinite(inputs).all().item())
            or rows.dtype != torch.int64
            or rows.ndim != 1
            or priorities.dtype != torch.int64
            or priorities.ndim != 1
            or int(rows.numel()) != int(inputs.shape[0])
            or int(priorities.numel()) != int(inputs.shape[0])
            or int(rows.numel()) != expected_cardinality
        ):
            raise SampleParallelMergeError(
                f"published activation-cache tensor contract differs for {qname}"
            )
        if rows.numel() and (
            int(rows.min().item()) < 0
            or int(rows.max().item()) >= total_candidate_rows
        ):
            raise SampleParallelMergeError(
                f"published activation-cache global row domain differs for {qname}"
            )
        if int(torch.unique(rows).numel()) != int(rows.numel()):
            raise SampleParallelMergeError(
                f"published activation-cache global rows are not unique for {qname}"
            )
        observed_rows_np = rows.detach().to("cpu").contiguous().numpy()
        observed_priorities_np = priorities.detach().to(
            "cpu"
        ).contiguous().numpy()
        recomputed_priorities_np = _reference_activation_priorities_numpy(
            str(cover["global_calibration_hash"]), qname, observed_rows_np
        )
        if not np.array_equal(
            observed_priorities_np, recomputed_priorities_np
        ):
            raise SampleParallelMergeError(
                f"published activation-cache priorities differ from global "
                f"row identities for {qname}"
            )
        priority_group = activation_priority_group(qname)
        prior_group_rows = observed_rows_by_group.get(priority_group)
        if prior_group_rows is None:
            observed_rows_by_group[priority_group] = rows.clone()
        elif not torch.equal(prior_group_rows, rows):
            raise SampleParallelMergeError(
                f"published activation-cache fused siblings in "
                f"{priority_group!r} do not retain identical global rows"
            )
        expected_top_r = expected_top_r_by_group.get(priority_group)
        if expected_top_r is None:
            expected_top_r = _reference_top_r_global_domain_numpy(
                str(cover["global_calibration_hash"]),
                qname,
                total_candidate_rows,
                declared_max_rows,
            )
            expected_top_r_by_group[priority_group] = expected_top_r
        expected_rows_np, expected_priorities_np = expected_top_r
        if not np.array_equal(observed_rows_np, expected_rows_np):
            raise SampleParallelMergeError(
                f"published activation-cache rows are not the canonical "
                f"exact global top-R for {qname}"
            )
        if not np.array_equal(
            observed_priorities_np, expected_priorities_np
        ):
            raise SampleParallelMergeError(
                f"published activation-cache rows are not in canonical "
                f"global priority order for {qname}"
            )
        identity = {
            "inputs": _tensor_identity(inputs),
            "row_indices": _tensor_identity(rows),
            "row_priorities": _tensor_identity(priorities),
            "source_shards": contributing,
            "priority_group": activation_priority_group(qname),
        }
        identity["identity_sha256"] = canonical_json_sha256(
            identity, where=f"published sample-parallel activation {qname}"
        )
        replayed_records[qname] = identity
        if records[qname] != identity or payload.get("name") != qname:
            raise SampleParallelMergeError(
                f"published activation-cache identity differs for {qname}"
            )
        stamp = payload.get("sample_parallel_merge")
        if not isinstance(stamp, Mapping) or stamp != {
            "schema": ACTIVATION_CACHE_MERGE_SCHEMA,
            "cover_identity_sha256": cover_digest,
            "identity_sha256": identity["identity_sha256"],
            "priority_schema": ACTIVATION_PRIORITY_SCHEMA,
            "priority_group": activation_priority_group(qname),
        }:
            raise SampleParallelMergeError(
                f"published activation-cache merge stamp differs for {qname}"
            )
    replayed_manifest_body: dict[str, object] = {
        "schema": ACTIVATION_CACHE_MERGE_SCHEMA,
        "cover": cover,
        "cover_identity_sha256": cover_digest,
        "exact_disjoint_cover": True,
        "exact_global_priority_top_r": True,
        "local_global_top_r_associativity_verified": True,
        "independent_numpy_priority_verifier": True,
        "priority_schema": ACTIVATION_PRIORITY_SCHEMA,
        "supported_activation_layout": "dense_linear",
        "routed_packed_activation_caches_rejected": True,
        "fused_group_alignment_verified": True,
        "qnames": len(replayed_records),
        "max_rows": declared_max_rows,
        "records": replayed_records,
        "activation_qname_manifest_sha256": activation_manifest[
            "identity_sha256"
        ],
        "source_census_sha256": cover["qname_census"][
            "source_census"
        ]["identity_sha256"],
    }
    if body != replayed_manifest_body:
        raise SampleParallelMergeError(
            "published activation-cache merge manifest replay differs"
        )
    return dict(manifest)


__all__ = [
    "ACTIVATION_CACHE_MANIFEST",
    "ACTIVATION_CACHE_MERGE_SCHEMA",
    "ACTIVATION_CACHE_SHARD_SCHEMA",
    "ACTIVATION_PRIORITY_SCHEMA",
    "IMPORTANCE_NORMALIZATION_SCHEMA",
    "SAMPLE_PARALLEL_COVER_SCHEMA",
    "SAMPLE_PARALLEL_MERGE_SCHEMA",
    "SAMPLE_PARALLEL_SHARD_SCHEMA",
    "SampleParallelMergeError",
    "activation_priority_group",
    "activation_row_priorities",
    "build_sample_parallel_cover",
    "merge_sample_parallel_activation_caches",
    "validate_merged_activation_cache_output",
    "merge_sample_parallel_probe_payloads",
    "validate_sample_parallel_stat_row",
    "validate_sample_parallel_merged_stat_row",
]
