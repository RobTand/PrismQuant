"""Sample-parallel calibration and incremental-probe producer identities.

Each worker loads the same immutable tokenized calibration tensor, selects one
deterministic contiguous sample partition, and runs the ordinary incremental
probe over the complete qname scope.  The strict merger lives in
``sample_parallel_probe_merge``; keeping it separate makes this module solely
responsible for immutable calibration, two-stage importance normalization, and
worker-count-invariant activation-row selection.
"""
from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any

import numpy as np
import torch

from prismaquant.perturbed_x_cache import calibration_data_hash
from prismaquant.sample_parallel_probe_contract import (
    ACTIVATION_CACHE_SHARD_SCHEMA,
    ACTIVATION_PRIORITY_MAX_ROWS,
    ACTIVATION_PRIORITY_SCHEMA,
    ROW_INDICES_SCOPE,
    activation_priority_group,
    activation_row_priorities,
    validate_activation_priority_domain,
)
from prismaquant.sensitivity_probe import load_calibration


CALIBRATION_SCHEMA = "prismaquant.sample_parallel_probe.calibration.v1"
PARTITION_SCHEMA = "prismaquant.sample_parallel_probe.partition.v1"
LOCAL_IMPORTANCE_SCHEMA = (
    "prismaquant.sample_parallel_probe.local_importance_stats.v2"
)
GLOBAL_IMPORTANCE_SCHEMA = (
    "prismaquant.sample_parallel_probe.global_importance_normalization.v2"
)
IMPORTANCE_EXECUTION_BINDING_SCHEMA = (
    "prismaquant.sample_parallel_probe.importance_execution_binding.v1"
)
ACTIVATION_SCOPE_SCHEMA = (
    "prismaquant.sample_parallel_probe.activation_scope.v1"
)
# The schema string keeps its original spelling on purpose: it is the wire
# identifier stamped into every run contract and census already on disk.  The
# producer that minted it (the Gridbook lane's RTX4090 campaign) retired
# 2026-09-02 -- see archive/gridbook_lane_2026-09-02/ -- but renaming the
# identifier would silently invalidate that recorded evidence.
SOURCE_QNAME_CENSUS_SCHEMA = (
    "prismaquant.sample_parallel_probe.rtx4090_source_qname_census.v2"
)
SOURCE_CENSUS_STABLE_PROJECTION_SCHEMA = (
    "prismaquant.sample_parallel_probe.worker_source_projection.v2"
)
SOURCE_MODEL_CONTENT_SCHEMA = (
    "prismaquant.sample_parallel_probe.model_content.v2"
)
QNAME_MANIFEST_SCHEMA = (
    "prismaquant.sample_parallel_probe.qname_manifest.v1"
)
EXECUTION_IDENTITY_SCHEMA = (
    "prismaquant.sample_parallel_probe.execution_identity.v1"
)
RUN_CONTRACT_SCHEMA = "prismaquant.sample_parallel_probe.run_contract.v1"
MERGE_BUNDLE_COMMIT_SCHEMA = (
    "prismaquant.sample_parallel_probe.merge_bundle_commit.v1"
)
WORKER_SOURCE_CACHE_RECEIPT_SCHEMA = (
    "prismaquant.sample_parallel_probe.worker_source_cache_receipt.v1"
)
MERGE_BUNDLE_PROBE = "probe.pkl"
MERGE_BUNDLE_ACTIVATIONS = "activation_cache"
MERGE_BUNDLE_COMMIT = "commit.json"

BODY_STATS_AND_ACTIVATION = "dense_body_stats_and_activation"
LM_HEAD_STATS_ONLY = "terminal_bf16_lm_head_stats_only"
MTP_STATS_ONLY = "terminal_bf16_mtp_stats_only"
TEXT_EXCLUDED = "excluded_text_only_source_linear"

BODY_TOKEN_ROWS = "seqlen"
LM_HEAD_TOKEN_ROWS = "seqlen_minus_1"
MTP_TOKEN_ROWS = "seqlen_minus_2"

IMPORTANCE_ENABLED_CONTRACT = (
    "two_stage_global_ce_mean_source_and_producer_bound_v2"
)
IMPORTANCE_DISABLED_CONTRACT = "disabled_v1"
MARGINAL_ENABLED_CONTRACT = "dense_fp32_raw_sum_and_absmax_v1"
MARGINAL_DISABLED_CONTRACT = "disabled_v1"
ESTIMATOR_CONTRACT = "dense_empirical_ce_fisher_v1"
MATH_CONTRACT = {
    "raw_accumulation": "local_fp32_nonnegative_sums_v1",
    "fisher_finalization": "once_after_global_sample_union_v1",
    "fisher_denominator": "global_nsamples_times_seqlen_v1",
    "body_token_cover": BODY_TOKEN_ROWS,
    "lm_head_token_cover": LM_HEAD_TOKEN_ROWS,
    "mtp_token_cover": MTP_TOKEN_ROWS,
    "ce_logits_dtype": "fp32",
}

_PARTITION_KEYS = frozenset({
    "schema", "global_calibration_hash", "calibration_artifact_sha256",
    "global_samples", "seqlen", "partition_count", "partition_index",
    "sample_indices", "sample_start", "sample_stop",
    "local_calibration_hash", "local_samples", "dataset", "model",
    "calib_seed",
})
_LOCAL_IMPORTANCE_KEYS = frozenset({
    "schema", "partition", "execution_identity_sha256", "ce_sum",
    "ce_count", "barrier_execution",
    "phase1_reused_across_barrier", "expected_apply_overhead",
    "receipt_sha256",
})
_EXECUTION_IDENTITY_KEYS = frozenset({
    "schema", "model", "dataset", "calib_seed", "dtype",
    "calibration_modality", "importance_weighting", "importance_contract",
    "probe_schedule", "include_visual",
    "activation_scope", "activation_rows_limit", "emit_marginals",
    "marginal_contract", "estimator_contract", "math_contract", "h_detail",
    "routed_packed_stats", "source_census_sha256",
    "model_content_sha256",
    "probe_qname_manifest_sha256", "activation_qname_manifest_sha256",
    "terminal_qname_manifest_sha256", "identity_sha256",
    "producer_snapshot_sha256", "container_image_digest",
    "producer_snapshot_commit", "producer_snapshot_tree",
})


class SampleParallelProbeError(ValueError):
    """A calibration partition, sufficient statistic, or cache is invalid."""


def _strict_json_loads(payload: str, *, where: str) -> object:
    """Decode one trusted JSON artifact while rejecting duplicate members.

    Python's ordinary JSON decoder silently keeps the last value for a
    duplicate object key.  Contract digests and closed schemas must instead
    have one unambiguous parse at every nesting level.
    """
    def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SampleParallelProbeError(
                    f"{where} contains duplicate JSON member {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=_object_from_pairs)
    except SampleParallelProbeError:
        raise
    except (TypeError, ValueError) as exc:
        raise SampleParallelProbeError(f"{where} is invalid JSON") from exc


def _canonical_sha256(value: object, *, where: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SampleParallelProbeError(
            f"{where} is not canonical JSON data"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes_no_clobber(path: Path, payload: bytes) -> None:
    """Durably publish one new file without replacing a concurrent writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SampleParallelProbeError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-link publication is atomic and fails with EEXIST.  The
            # temporary and destination are deliberately on one filesystem.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SampleParallelProbeError(
                f"refusing to overwrite artifact: {path}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_torch_artifact_no_clobber(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SampleParallelProbeError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SampleParallelProbeError(
                f"refusing to overwrite artifact: {path}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class SamplePartition:
    partition_index: int
    partition_count: int
    global_samples: int
    sample_start: int
    sample_stop: int
    sample_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.partition_count < 1:
            raise SampleParallelProbeError("partition_count must be positive")
        if self.partition_index not in range(self.partition_count):
            raise SampleParallelProbeError("partition_index is out of range")
        if self.global_samples < self.partition_count:
            raise SampleParallelProbeError(
                "global sample count must be at least the partition count"
            )
        expected = tuple(range(self.sample_start, self.sample_stop))
        if self.sample_indices != expected or not expected:
            raise SampleParallelProbeError(
                "sample partition must be one nonempty contiguous range"
            )


def plan_sample_partitions(
    global_samples: int, partition_count: int,
) -> tuple[SamplePartition, ...]:
    """Balanced deterministic contiguous ranges covering ``range(N)``."""
    n_samples = int(global_samples)
    n_parts = int(partition_count)
    if n_samples < 1 or n_parts < 1 or n_parts > n_samples:
        raise SampleParallelProbeError(
            "need 1 <= partition_count <= global_samples"
        )
    base, remainder = divmod(n_samples, n_parts)
    parts: list[SamplePartition] = []
    start = 0
    for index in range(n_parts):
        size = base + (1 if index < remainder else 0)
        stop = start + size
        parts.append(SamplePartition(
            partition_index=index,
            partition_count=n_parts,
            global_samples=n_samples,
            sample_start=start,
            sample_stop=stop,
            sample_indices=tuple(range(start, stop)),
        ))
        start = stop
    if start != n_samples:
        raise AssertionError("sample partition planner lost samples")
    return tuple(parts)


def _normalize_partition_contract(
    raw: Mapping[str, object], *, where: str,
) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != _PARTITION_KEYS:
        raise SampleParallelProbeError(
            f"{where} fields differ from the closed v1 partition contract"
        )
    integer_fields = (
        "global_samples", "seqlen", "partition_count", "partition_index",
        "sample_start", "sample_stop", "local_samples", "calib_seed",
    )
    string_fields = (
        "schema", "global_calibration_hash", "calibration_artifact_sha256",
        "local_calibration_hash", "dataset", "model",
    )
    if (
        any(type(raw[field]) is not int for field in integer_fields)
        or any(type(raw[field]) is not str for field in string_fields)
        or not isinstance(raw["sample_indices"], list)
        or any(type(value) is not int for value in raw["sample_indices"])
    ):
        raise SampleParallelProbeError(f"{where} field types differ")
    try:
        row: dict[str, object] = {
            "schema": str(raw["schema"]),
            "global_calibration_hash": str(raw["global_calibration_hash"]),
            "calibration_artifact_sha256": str(
                raw["calibration_artifact_sha256"]
            ),
            "global_samples": int(raw["global_samples"]),
            "seqlen": int(raw["seqlen"]),
            "partition_count": int(raw["partition_count"]),
            "partition_index": int(raw["partition_index"]),
            "sample_indices": [int(value) for value in raw["sample_indices"]],
            "sample_start": int(raw["sample_start"]),
            "sample_stop": int(raw["sample_stop"]),
            "local_calibration_hash": str(raw["local_calibration_hash"]),
            "local_samples": int(raw["local_samples"]),
            "dataset": str(raw["dataset"]),
            "model": str(raw["model"]),
            "calib_seed": int(raw["calib_seed"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SampleParallelProbeError(f"{where} is malformed") from exc
    if row["schema"] != PARTITION_SCHEMA:
        raise SampleParallelProbeError(f"{where} schema differs")
    if not re.fullmatch(
        r"[0-9a-f]{32}", str(row["global_calibration_hash"])
    ) or not re.fullmatch(
        r"[0-9a-f]{32}", str(row["local_calibration_hash"])
    ):
        raise SampleParallelProbeError(f"{where} calibration hash is invalid")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(row["calibration_artifact_sha256"])
    ):
        raise SampleParallelProbeError(f"{where} artifact SHA-256 is invalid")
    start = int(row["sample_start"])
    stop = int(row["sample_stop"])
    indices = row["sample_indices"]
    if (
        int(row["global_samples"]) < 1
        or int(row["seqlen"]) < 3
        or int(row["partition_count"]) < 1
        or int(row["partition_index"]) not in range(int(row["partition_count"]))
        or start < 0
        or stop <= start
        or indices != list(range(start, stop))
        or int(row["local_samples"]) != stop - start
        or stop > int(row["global_samples"])
        or not row["dataset"]
        or not row["model"]
        or int(row["calib_seed"]) < 0
    ):
        raise SampleParallelProbeError(f"{where} range/provenance is invalid")
    try:
        validate_activation_priority_domain(
            int(row["global_samples"]), int(row["seqlen"])
        )
    except ValueError as exc:
        raise SampleParallelProbeError(
            f"{where} activation priority domain is invalid"
        ) from exc
    return row


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opened_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    """Process-local identity used to detect mutation during one read."""
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_nlink), int(value.st_size), int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _open_regular_bytes_at(
    directory_fd: int,
    name: str,
    *,
    where: str,
) -> tuple[bytes, str, tuple[int, ...]]:
    """Read and hash one regular child through the same no-follow fd."""
    if not name or name != Path(name).name:
        raise SampleParallelProbeError(f"{where} member name is unsafe")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise SampleParallelProbeError(
            f"{where} cannot be safely opened"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SampleParallelProbeError(f"{where} is not a regular file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _opened_stat_identity(after) != _opened_stat_identity(before):
            raise SampleParallelProbeError(
                f"{where} changed while it was being consumed"
            )
        return (
            b"".join(chunks), digest.hexdigest(),
            _opened_stat_identity(after),
        )
    finally:
        os.close(descriptor)


def _open_directory_nofollow(path: Path, *, where: str) -> int:
    """Open one real directory and prove the fd names the inspected inode."""
    try:
        before = path.lstat()
    except OSError as exc:
        raise SampleParallelProbeError(f"{where} is unreadable") from exc
    if not stat.S_ISDIR(before.st_mode) or path.is_symlink():
        raise SampleParallelProbeError(f"{where} is not a regular directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SampleParallelProbeError(
            f"{where} cannot be safely opened"
        ) from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or int(opened.st_dev) != int(before.st_dev)
        or int(opened.st_ino) != int(before.st_ino)
    ):
        os.close(descriptor)
        raise SampleParallelProbeError(f"{where} changed while opening")
    return descriptor


def prepare_worker_source_cache(
    *,
    model: str | Path,
    output: str | Path,
    offload_folder: str | Path,
) -> dict[str, object]:
    """Create one host-local complete streamed-model identity cache.

    The cache belongs to :mod:`prismaquant.cost_streaming`; this command is
    only the missing two-host bootstrap around that existing owner.  A fresh
    value is built below a private same-filesystem staging directory, fully
    validated against this host's checkpoint, and hard-linked into place so a
    concurrent or stale destination can never be replaced.  An existing file
    is reusable only when the ordinary complete-checkpoint validator accepts
    it exactly.
    """
    from prismaquant.cost_streaming import (
        build_streamed_causal_lm,
        build_streamed_model_identity,
        compact_streamed_model_identity,
        validate_cached_streamed_model_identity,
    )
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.model_profiles import detect_profile

    source = Path(model)
    if not source.is_absolute() or source.is_symlink():
        raise SampleParallelProbeError(
            "worker source model must be one absolute non-symlink path"
        )
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise SampleParallelProbeError(
            "worker source model is unreadable"
        ) from exc
    if not source.is_dir():
        raise SampleParallelProbeError(
            "worker source model must be one local checkpoint directory"
        )

    destination = Path(output)
    if not destination.is_absolute() or destination.is_symlink():
        raise SampleParallelProbeError(
            "worker source cache output must be absolute and non-symlink"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.parent.resolve(strict=True) / destination.name

    def _receipt(
        identity: Mapping[str, object], *, disposition: str,
    ) -> dict[str, object]:
        compact = compact_streamed_model_identity(
            identity, where="sample-parallel worker source cache"
        )
        return {
            "schema": WORKER_SOURCE_CACHE_RECEIPT_SCHEMA,
            "disposition": disposition,
            "model": str(source),
            "cache": str(destination),
            "cache_sha256": _sha256_file(destination),
            "identity": compact,
        }

    if destination.exists():
        if not destination.is_file():
            raise SampleParallelProbeError(
                "worker source cache destination is not a regular file; "
                "refusing overwrite"
            )
        try:
            existing = validate_cached_streamed_model_identity(
                source, destination, require_complete_checkpoint=True,
            )
        except Exception as exc:
            raise SampleParallelProbeError(
                "existing worker source cache is not an exact complete "
                "checkpoint identity; refusing overwrite"
            ) from exc
        return _receipt(existing, disposition="validated_reuse")

    scratch = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.build-", dir=destination.parent,
    ))
    staged = scratch / "streamed_model_identity_cache.json"
    runner = None
    try:
        device = require_cuda_hot_path(
            "sample_parallel_probe.prepare_worker_source_cache", "cuda"
        )
        profile = detect_profile(str(source))
        offload = Path(offload_folder)
        if not offload.is_absolute() or offload.is_symlink():
            raise SampleParallelProbeError(
                "worker source-cache offload folder must be absolute and "
                "non-symlink"
            )
        offload.mkdir(parents=True, exist_ok=True)
        offload = offload.resolve(strict=True)
        runner = build_streamed_causal_lm(
            str(source),
            device=device,
            dtype=torch.bfloat16,
            offload_folder=str(offload),
            profile=profile,
            max_cache_slots=1,
            prefetch_lookahead=0,
        )
        build_streamed_model_identity(
            runner, str(source), identity_cache_path=staged,
        )
        validated = validate_cached_streamed_model_identity(
            source, staged, require_complete_checkpoint=True,
        )
        try:
            os.link(staged, destination)
        except FileExistsError as exc:
            raise SampleParallelProbeError(
                f"refusing to overwrite worker source cache: {destination}"
            ) from exc
        _fsync_directory(destination.parent)
        # Revalidate the published inode rather than trusting the staging name.
        validated = validate_cached_streamed_model_identity(
            source, destination, require_complete_checkpoint=True,
        )
        return _receipt(validated, disposition="created")
    except SampleParallelProbeError:
        raise
    except Exception as exc:
        raise SampleParallelProbeError(
            f"worker source cache preparation failed: {exc}"
        ) from exc
    finally:
        if runner is not None:
            runner.shutdown()
        shutil.rmtree(scratch, ignore_errors=True)


def prepare_global_calibration(
    *,
    model: str,
    dataset: str,
    nsamples: int,
    seqlen: int,
    calib_seed: int,
    partition_count: int,
    output: str | Path,
    manifest_output: str | Path,
) -> dict[str, object]:
    """Tokenize once and publish one immutable global calibration artifact."""
    output_path = Path(output)
    manifest_path = Path(manifest_output)
    if output_path == manifest_path:
        raise SampleParallelProbeError(
            "calibration artifact and manifest outputs must differ"
        )
    if manifest_path.exists() and not output_path.exists():
        raise SampleParallelProbeError(
            "calibration manifest exists without its artifact; refusing to "
            "publish a mismatched pair"
        )
    artifact: Mapping[str, object] | None = None
    if output_path.exists():
        try:
            loaded = torch.load(
                output_path, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            raise SampleParallelProbeError(
                "existing calibration artifact is unreadable; refusing overwrite"
            ) from exc
        try:
            artifact = _validate_calibration_artifact(loaded)
        except SampleParallelProbeError as exc:
            raise SampleParallelProbeError(
                "existing calibration artifact is malformed; refusing overwrite"
            ) from exc
        ids = artifact["ids"]
    else:
        from transformers import AutoTokenizer
        from prismaquant.sensitivity_probe import stage_text_only

        tokenizer = AutoTokenizer.from_pretrained(
            stage_text_only(model), trust_remote_code=True,
            local_files_only=Path(model).exists(),
        )
        ids = load_calibration(
            tokenizer, dataset, int(nsamples), int(seqlen),
            calib_seed=int(calib_seed),
        ).detach().to("cpu", dtype=torch.long).contiguous()
    if ids.ndim != 2 or tuple(ids.shape) != (int(nsamples), int(seqlen)):
        raise SampleParallelProbeError(
            f"tokenized calibration shape {tuple(ids.shape)} differs from "
            f"({nsamples}, {seqlen})"
        )
    try:
        validate_activation_priority_domain(ids.size(0), ids.size(1))
    except ValueError as exc:
        raise SampleParallelProbeError(
            "global calibration exceeds the activation priority uint32 domain"
        ) from exc
    global_hash = calibration_data_hash(ids)
    partitions = plan_sample_partitions(ids.size(0), int(partition_count))
    partition_rows = [
        {**asdict(part), "sample_indices": list(part.sample_indices)}
        for part in partitions
    ]
    expected_artifact = {
        "schema": CALIBRATION_SCHEMA,
        "ids": ids,
        "global_calibration_hash": global_hash,
        "global_samples": int(ids.size(0)),
        "seqlen": int(ids.size(1)),
        "calib_seed": int(calib_seed),
        "dataset": str(dataset),
        "model": str(model),
        "partition_count": int(partition_count),
        "partitions": partition_rows,
    }
    if artifact is None:
        _publish_torch_artifact_no_clobber(output_path, expected_artifact)
    else:
        comparable = dict(artifact)
        comparable["ids"] = ids
        if set(comparable) != set(expected_artifact) or any(
            comparable[key] != expected_artifact[key]
            for key in expected_artifact if key != "ids"
        ) or not torch.equal(ids, expected_artifact["ids"]):
            raise SampleParallelProbeError(
                "existing calibration artifact differs from the requested "
                "deterministic run; refusing overwrite"
            )
    artifact_sha256 = _sha256_file(output_path)
    partition_contracts = []
    for part in partitions:
        local = ids[part.sample_start:part.sample_stop].contiguous()
        partition_contracts.append({
            "schema": PARTITION_SCHEMA,
            "global_calibration_hash": global_hash,
            "calibration_artifact_sha256": artifact_sha256,
            "global_samples": int(ids.size(0)),
            "seqlen": int(ids.size(1)),
            "partition_count": part.partition_count,
            "partition_index": part.partition_index,
            "sample_indices": list(part.sample_indices),
            "sample_start": part.sample_start,
            "sample_stop": part.sample_stop,
            "local_calibration_hash": calibration_data_hash(local),
            "local_samples": int(local.size(0)),
            "dataset": str(dataset),
            "model": str(model),
            "calib_seed": int(calib_seed),
        })
    manifest = {
        "schema": CALIBRATION_SCHEMA,
        "calibration_artifact_sha256": artifact_sha256,
        "global_calibration_hash": global_hash,
        "global_samples": int(ids.size(0)),
        "seqlen": int(ids.size(1)),
        "calib_seed": int(calib_seed),
        "dataset": str(dataset),
        "model": str(model),
        "partition_count": int(partition_count),
        "partitions": partition_rows,
        "partition_contracts": partition_contracts,
    }
    manifest = validate_calibration_manifest(manifest)
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    if manifest_path.exists():
        try:
            existing_manifest = _strict_json_loads(
                manifest_path.read_text(encoding="utf-8"),
                where="existing calibration manifest",
            )
        except SampleParallelProbeError:
            raise
        except (OSError, ValueError) as exc:
            raise SampleParallelProbeError(
                "existing calibration manifest is unreadable; refusing overwrite"
            ) from exc
        if existing_manifest != manifest:
            raise SampleParallelProbeError(
                "existing calibration manifest differs; refusing overwrite"
            )
    else:
        _atomic_write_bytes_no_clobber(manifest_path, manifest_bytes)
    return manifest


def _validate_calibration_artifact(
    raw: object,
) -> dict[str, object]:
    keys = {
        "schema", "ids", "global_calibration_hash", "global_samples",
        "seqlen", "calib_seed", "dataset", "model", "partition_count",
        "partitions",
    }
    if not isinstance(raw, Mapping) or set(raw) != keys or raw.get(
        "schema"
    ) != CALIBRATION_SCHEMA:
        raise SampleParallelProbeError("global calibration artifact schema differs")
    if any(type(raw.get(field)) is not int for field in (
        "global_samples", "seqlen", "calib_seed", "partition_count",
    )) or any(type(raw.get(field)) is not str for field in (
        "global_calibration_hash", "dataset", "model",
    )):
        raise SampleParallelProbeError(
            "global calibration artifact scalar types differ"
        )
    ids = raw.get("ids")
    if (
        not isinstance(ids, torch.Tensor)
        or ids.ndim != 2
        or ids.dtype != torch.int64
    ):
        raise SampleParallelProbeError(
            "global calibration ids are not a 2-D int64 tensor"
        )
    ids = ids.detach().to("cpu").contiguous()
    if (
        tuple(ids.shape) != (raw["global_samples"], raw["seqlen"])
        or raw["global_samples"] < 1
        or raw["seqlen"] < 3
        or raw["calib_seed"] < 0
        or raw["partition_count"] < 1
        or not raw["dataset"]
        or not raw["model"]
        or re.fullmatch(
            r"[0-9a-f]{32}", raw["global_calibration_hash"]
        ) is None
        or calibration_data_hash(ids) != raw["global_calibration_hash"]
    ):
        raise SampleParallelProbeError(
            "global calibration artifact content/provenance differs"
        )
    try:
        validate_activation_priority_domain(ids.size(0), ids.size(1))
        partitions = plan_sample_partitions(
            ids.size(0), raw["partition_count"]
        )
    except (TypeError, ValueError) as exc:
        raise SampleParallelProbeError(
            "global calibration artifact partition domain differs"
        ) from exc
    declared = raw.get("partitions")
    partition_keys = {
        "partition_index", "partition_count", "global_samples",
        "sample_start", "sample_stop", "sample_indices",
    }
    if not isinstance(declared, list) or any(
        not isinstance(row, Mapping)
        or set(row) != partition_keys
        or any(type(row.get(field)) is not int for field in (
            "partition_index", "partition_count", "global_samples",
            "sample_start", "sample_stop",
        ))
        or not isinstance(row.get("sample_indices"), list)
        or any(type(value) is not int for value in row["sample_indices"])
        for row in declared
    ):
        raise SampleParallelProbeError(
            "global calibration artifact partition fields differ"
        )
    expected = [
        {**asdict(item), "sample_indices": list(item.sample_indices)}
        for item in partitions
    ]
    if declared != expected:
        raise SampleParallelProbeError("stored sample partition plan differs")
    return {**dict(raw), "ids": ids}


def load_calibration_partition(
    artifact_path: str | Path, *, partition_index: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Load the immutable global tensor once, verify it, then take one slice."""
    try:
        artifact = torch.load(
            artifact_path, map_location="cpu", weights_only=False,
        )
    except Exception as exc:
        raise SampleParallelProbeError(
            f"global calibration artifact is unreadable: {artifact_path}"
        ) from exc
    artifact = _validate_calibration_artifact(artifact)
    ids = artifact["ids"]
    observed_hash = artifact["global_calibration_hash"]
    partitions = plan_sample_partitions(ids.size(0), artifact["partition_count"])
    if type(partition_index) is not int:
        raise SampleParallelProbeError("sample partition index type differs")
    index = partition_index
    if index not in range(len(partitions)):
        raise SampleParallelProbeError("sample partition index is out of range")
    part = partitions[index]
    local = ids[part.sample_start:part.sample_stop].clone().contiguous()
    contract: dict[str, object] = {
        "schema": PARTITION_SCHEMA,
        "global_calibration_hash": observed_hash,
        "calibration_artifact_sha256": _sha256_file(artifact_path),
        "global_samples": int(ids.size(0)),
        "seqlen": int(ids.size(1)),
        "partition_count": part.partition_count,
        "partition_index": part.partition_index,
        "sample_indices": list(part.sample_indices),
        "sample_start": part.sample_start,
        "sample_stop": part.sample_stop,
        "local_calibration_hash": calibration_data_hash(local),
        "local_samples": int(local.size(0)),
        "dataset": artifact["dataset"],
        "model": artifact["model"],
        "calib_seed": artifact["calib_seed"],
    }
    contract = _normalize_partition_contract(
        contract, where="loaded calibration partition"
    )
    return local, contract


def validate_calibration_manifest(raw: Mapping[str, object]) -> dict[str, object]:
    keys = {
        "schema", "calibration_artifact_sha256",
        "global_calibration_hash", "global_samples", "seqlen", "calib_seed",
        "dataset", "model", "partition_count", "partitions",
        "partition_contracts",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != keys
        or raw.get("schema") != CALIBRATION_SCHEMA
    ):
        raise SampleParallelProbeError(
            "global calibration manifest fields/schema differ"
        )
    if any(type(raw.get(field)) is not int for field in (
        "global_samples", "seqlen", "calib_seed", "partition_count",
    )) or any(type(raw.get(field)) is not str for field in (
        "calibration_artifact_sha256", "global_calibration_hash",
        "dataset", "model",
    )) or not isinstance(raw.get("partitions"), list):
        raise SampleParallelProbeError(
            "global calibration manifest field types differ"
        )
    contracts_raw = raw.get("partition_contracts")
    if not isinstance(contracts_raw, Sequence) or isinstance(
        contracts_raw, (str, bytes)
    ):
        raise SampleParallelProbeError(
            "global calibration manifest lacks partition contracts"
        )
    contracts, global_hash, global_samples, seqlen = _validate_partition_cover(
        contracts_raw  # type: ignore[arg-type]
    )
    first = contracts[0]
    expected_partitions = [{
        "partition_index": contract["partition_index"],
        "partition_count": contract["partition_count"],
        "global_samples": contract["global_samples"],
        "sample_start": contract["sample_start"],
        "sample_stop": contract["sample_stop"],
        "sample_indices": contract["sample_indices"],
    } for contract in contracts]
    if (
        raw.get("calibration_artifact_sha256")
        != first["calibration_artifact_sha256"]
        or raw.get("global_calibration_hash") != global_hash
        or raw.get("global_samples") != global_samples
        or raw.get("seqlen") != seqlen
        or raw.get("calib_seed") != first["calib_seed"]
        or raw.get("dataset") != first["dataset"]
        or raw.get("model") != first["model"]
        or raw.get("partition_count") != len(contracts)
        or raw.get("partitions") != expected_partitions
    ):
        raise SampleParallelProbeError(
            "global calibration manifest/partition invariant differs"
        )
    return {
        "schema": CALIBRATION_SCHEMA,
        "calibration_artifact_sha256": first["calibration_artifact_sha256"],
        "global_calibration_hash": global_hash,
        "global_samples": global_samples,
        "seqlen": seqlen,
        "calib_seed": first["calib_seed"],
        "dataset": first["dataset"],
        "model": first["model"],
        "partition_count": len(contracts),
        "partitions": expected_partitions,
        "partition_contracts": contracts,
    }


def write_local_importance_stats(
    path: str | Path,
    *,
    partition_contract: Mapping[str, object],
    execution_identity_sha256: str,
    ce_sum: float,
    ce_count: int,
) -> dict[str, object]:
    partition = _normalize_partition_contract(
        partition_contract, where="local importance partition"
    )
    value_sum = float(ce_sum)
    value_count = int(ce_count)
    if not np.isfinite(value_sum) or value_sum <= 0.0 or value_count <= 0:
        raise SampleParallelProbeError("local CE sufficient statistics are invalid")
    execution_digest = str(execution_identity_sha256)
    if re.fullmatch(r"[0-9a-f]{64}", execution_digest) is None:
        raise SampleParallelProbeError(
            "local CE execution identity SHA-256 is invalid"
        )
    expected_count = int(partition["local_samples"]) * (
        int(partition["seqlen"]) - 1
    )
    if value_count != expected_count:
        raise SampleParallelProbeError(
            "local CE count differs from the complete shifted-token cover: "
            f"observed={value_count}, expected={expected_count}"
        )
    body = {
        "schema": LOCAL_IMPORTANCE_SCHEMA,
        "partition": partition,
        "execution_identity_sha256": execution_digest,
        "ce_sum": value_sum,
        "ce_count": value_count,
        # v1 deliberately reruns phase 1 after the scalar barrier.  The
        # receipt says so instead of implying that forward activations were
        # reused across processes when they were not.
        "barrier_execution": "duplicate_phase1_forward_v1",
        "phase1_reused_across_barrier": False,
        "expected_apply_overhead": "one_additional_phase1_forward",
    }
    payload = {
        **body,
        "receipt_sha256": _canonical_sha256(
            body, where="local importance receipt"
        ),
    }
    _atomic_write_bytes_no_clobber(
        Path(path),
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return payload


def _validate_local_importance_payload(
    raw: Mapping[str, object], *, where: str,
) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != _LOCAL_IMPORTANCE_KEYS:
        raise SampleParallelProbeError(
            f"{where} fields differ from the closed v1 receipt"
        )
    payload = dict(raw)
    if payload.get("schema") != LOCAL_IMPORTANCE_SCHEMA:
        raise SampleParallelProbeError(f"{where} schema differs")
    digest = payload.pop("receipt_sha256", None)
    if digest != _canonical_sha256(payload, where=where):
        raise SampleParallelProbeError(f"{where} digest differs")
    partition = _normalize_partition_contract(
        payload.get("partition"), where=f"{where} partition"  # type: ignore[arg-type]
    )
    try:
        if type(payload["ce_count"]) is not int or type(
            payload["ce_sum"]
        ) not in (int, float):
            raise TypeError("CE statistic types differ")
        ce_sum = float(payload["ce_sum"])
        ce_count = int(payload["ce_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SampleParallelProbeError(f"{where} CE statistics differ") from exc
    expected_count = int(partition["local_samples"]) * (
        int(partition["seqlen"]) - 1
    )
    if not np.isfinite(ce_sum) or ce_sum <= 0.0 or ce_count != expected_count:
        raise SampleParallelProbeError(f"{where} CE statistics differ")
    if (
        payload.get("barrier_execution")
        != "duplicate_phase1_forward_v1"
        or payload.get("phase1_reused_across_barrier") is not False
        or payload.get("expected_apply_overhead")
        != "one_additional_phase1_forward"
    ):
        raise SampleParallelProbeError(f"{where} barrier provenance differs")
    execution_digest = payload.get("execution_identity_sha256")
    if (
        type(execution_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", execution_digest) is None
    ):
        raise SampleParallelProbeError(
            f"{where} execution identity differs"
        )
    return {
        **payload,
        "partition": partition,
        "ce_sum": ce_sum,
        "ce_count": ce_count,
        "receipt_sha256": str(digest),
    }


def load_local_importance_stats(
    path: str | Path,
    *,
    partition_contract: Mapping[str, object],
    execution_identity_sha256: str,
) -> dict[str, object]:
    """Validate a resumable local CE summary against this exact partition."""
    try:
        payload = _strict_json_loads(
            Path(path).read_text(encoding="utf-8"),
            where="local importance receipt",
        )
    except SampleParallelProbeError:
        raise
    except (OSError, ValueError) as exc:
        raise SampleParallelProbeError(
            f"local importance receipt is unreadable: {path}"
        ) from exc
    validated = _validate_local_importance_payload(
        payload, where="local importance receipt"
    )
    partition = _normalize_partition_contract(
        partition_contract, where="expected local importance partition"
    )
    if validated.get("partition") != partition:
        raise SampleParallelProbeError("local importance receipt partition differs")
    if validated.get("execution_identity_sha256") != str(
        execution_identity_sha256
    ):
        raise SampleParallelProbeError(
            "local importance receipt execution identity differs"
        )
    return validated


def merge_importance_stats(
    paths: Sequence[str | Path], output_path: str | Path,
) -> dict[str, object]:
    payloads = []
    for path in paths:
        try:
            payload = _strict_json_loads(
                Path(path).read_text(encoding="utf-8"),
                where=f"local importance receipt {path}",
            )
        except SampleParallelProbeError:
            raise
        except (OSError, ValueError) as exc:
            raise SampleParallelProbeError(
                f"local importance receipt is unreadable: {path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise SampleParallelProbeError("local importance receipt is malformed")
        payloads.append(_validate_local_importance_payload(
            payload, where=f"local importance receipt {path}"
        ))
    contracts, global_hash, global_samples, seqlen = _validate_partition_cover(
        [payload["partition"] for payload in payloads]
    )
    by_index = {
        int(payload["partition"]["partition_index"]): payload
        for payload in payloads
    }
    ordered = [by_index[index] for index in range(len(contracts))]
    execution_digests = {
        str(payload["execution_identity_sha256"]) for payload in ordered
    }
    if len(execution_digests) != 1:
        raise SampleParallelProbeError(
            "local importance receipts use different execution identities"
        )
    execution_digest = next(iter(execution_digests))
    total = sum(float(payload["ce_sum"]) for payload in ordered)
    count = sum(int(payload["ce_count"]) for payload in ordered)
    expected_count = global_samples * (seqlen - 1)
    if count != expected_count:
        raise SampleParallelProbeError(
            "global CE count differs from the exact shifted-token sample cover: "
            f"observed={count}, expected={expected_count}"
        )
    mean = total / max(count, 1)
    if not np.isfinite(mean) or mean <= 0.0:
        raise SampleParallelProbeError("global CE mean is invalid")
    identity = {
        "schema": GLOBAL_IMPORTANCE_SCHEMA,
        "execution_identity_sha256": execution_digest,
        "body_mode": "global_ce_mean",
        "body_global_ce_mean": mean,
        "body_global_ce_sum": total,
        "body_global_ce_count": count,
        "global_calibration_hash": global_hash,
        "calibration_artifact_sha256": contracts[0][
            "calibration_artifact_sha256"
        ],
        "model": contracts[0]["model"],
        "dataset": contracts[0]["dataset"],
        "calib_seed": contracts[0]["calib_seed"],
        "global_samples": global_samples,
        "seqlen": seqlen,
        "partition_count": len(contracts),
        "exact_sample_cover": True,
        "mtp_mode": "per_sample_ce_mean",
        "equivalence": "mathematical_sample_cover_with_numeric_tolerance",
        "bitwise_monolithic_equivalence_claimed": False,
        "barrier_execution": "duplicate_phase1_forward_v1",
        "phase1_reused_across_barrier": False,
        "expected_apply_overhead": "one_additional_phase1_forward",
        "local_receipts": [{
            "partition_index": index,
            "partition": ordered[index]["partition"],
            "local_receipt_sha256": ordered[index]["receipt_sha256"],
            "execution_identity_sha256": ordered[index][
                "execution_identity_sha256"
            ],
            "ce_sum": float(ordered[index]["ce_sum"]),
            "ce_count": int(ordered[index]["ce_count"]),
            "barrier_execution": ordered[index]["barrier_execution"],
            "phase1_reused_across_barrier": ordered[index][
                "phase1_reused_across_barrier"
            ],
            "expected_apply_overhead": ordered[index][
                "expected_apply_overhead"
            ],
        } for index in range(len(ordered))],
    }
    receipt = {
        **identity,
        "receipt_sha256": _canonical_sha256(
            identity, where="global importance receipt"
        ),
    }
    _atomic_write_bytes_no_clobber(
        Path(output_path),
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return receipt


def validate_global_importance_receipt(
    raw: Mapping[str, object],
    *,
    partition_contract: Mapping[str, object] | None = None,
    execution_identity_sha256: str | None = None,
) -> dict[str, object]:
    """Validate every global/local CE invariant without projecting fields."""
    if not isinstance(raw, Mapping) or raw.get("schema") != GLOBAL_IMPORTANCE_SCHEMA:
        raise SampleParallelProbeError("global importance receipt schema differs")
    receipt = dict(raw)
    digest = receipt.pop("receipt_sha256", None)
    if digest != _canonical_sha256(
        receipt, where="global importance receipt"
    ):
        raise SampleParallelProbeError("global importance receipt digest differs")
    required = {
        "schema", "body_mode", "body_global_ce_mean", "body_global_ce_sum",
        "execution_identity_sha256",
        "body_global_ce_count", "global_calibration_hash",
        "calibration_artifact_sha256", "model", "dataset", "calib_seed",
        "global_samples", "seqlen", "partition_count", "exact_sample_cover",
        "mtp_mode", "equivalence", "bitwise_monolithic_equivalence_claimed",
        "barrier_execution", "phase1_reused_across_barrier",
        "expected_apply_overhead", "local_receipts",
    }
    if set(receipt) != required:
        raise SampleParallelProbeError(
            "global importance receipt fields differ from the closed v1 contract"
        )
    if (
        receipt.get("body_mode") != "global_ce_mean"
        or receipt.get("mtp_mode") != "per_sample_ce_mean"
        or receipt.get("exact_sample_cover") is not True
        or receipt.get("equivalence")
        != "mathematical_sample_cover_with_numeric_tolerance"
        or receipt.get("bitwise_monolithic_equivalence_claimed") is not False
        or receipt.get("barrier_execution")
        != "duplicate_phase1_forward_v1"
        or receipt.get("phase1_reused_across_barrier") is not False
        or receipt.get("expected_apply_overhead")
        != "one_additional_phase1_forward"
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(receipt.get("execution_identity_sha256", "")),
        ) is None
    ):
        raise SampleParallelProbeError(
            "global importance receipt execution provenance differs"
        )
    if (
        any(type(receipt.get(key)) is not int for key in (
            "body_global_ce_count", "calib_seed", "global_samples", "seqlen",
            "partition_count",
        ))
        or any(type(receipt.get(key)) not in (int, float) for key in (
            "body_global_ce_mean", "body_global_ce_sum",
        ))
        or any(type(receipt.get(key)) is not str for key in (
            "global_calibration_hash", "calibration_artifact_sha256",
            "model", "dataset",
        ))
    ):
        raise SampleParallelProbeError(
            "global importance receipt field types differ"
        )
    locals_raw = receipt.get("local_receipts")
    if not isinstance(locals_raw, Sequence) or isinstance(
        locals_raw, (str, bytes)
    ) or not locals_raw:
        raise SampleParallelProbeError("global importance local receipts differ")
    local_rows: list[dict[str, object]] = []
    local_contracts: list[dict[str, object]] = []
    local_keys = {
        "partition_index", "partition", "local_receipt_sha256", "ce_sum",
        "execution_identity_sha256",
        "ce_count", "barrier_execution", "phase1_reused_across_barrier",
        "expected_apply_overhead",
    }
    for index, item in enumerate(locals_raw):
        if not isinstance(item, Mapping) or set(item) != local_keys:
            raise SampleParallelProbeError(
                "global importance local receipt fields differ"
            )
        if (
            type(item.get("partition_index")) is not int
            or type(item.get("ce_count")) is not int
            or type(item.get("ce_sum")) not in (int, float)
            or type(item.get("local_receipt_sha256")) is not str
            or type(item.get("execution_identity_sha256")) is not str
        ):
            raise SampleParallelProbeError(
                "global importance local receipt field types differ"
            )
        partition = _normalize_partition_contract(
            item.get("partition"),  # type: ignore[arg-type]
            where=f"global importance local partition {index}",
        )
        if int(item.get("partition_index", -1)) != int(
            partition["partition_index"]
        ):
            raise SampleParallelProbeError(
                "global importance local receipt index differs"
            )
        local_body = {
            "schema": LOCAL_IMPORTANCE_SCHEMA,
            "partition": partition,
            "execution_identity_sha256": item.get(
                "execution_identity_sha256"
            ),
            "ce_sum": item.get("ce_sum"),
            "ce_count": item.get("ce_count"),
            "barrier_execution": item.get("barrier_execution"),
            "phase1_reused_across_barrier": item.get(
                "phase1_reused_across_barrier"
            ),
            "expected_apply_overhead": item.get("expected_apply_overhead"),
            "receipt_sha256": item.get("local_receipt_sha256"),
        }
        validated = _validate_local_importance_payload(
            local_body, where=f"global importance local receipt {index}"
        )
        local_rows.append(validated)
        local_contracts.append(partition)
    contracts, global_hash, global_samples, seqlen = _validate_partition_cover(
        local_contracts
    )
    ordered = sorted(
        local_rows, key=lambda row: int(row["partition"]["partition_index"])
    )
    total = sum(float(row["ce_sum"]) for row in ordered)
    count = sum(int(row["ce_count"]) for row in ordered)
    mean = total / count
    if (
        receipt.get("global_calibration_hash") != global_hash
        or receipt.get("calibration_artifact_sha256")
        != contracts[0]["calibration_artifact_sha256"]
        or receipt.get("model") != contracts[0]["model"]
        or receipt.get("dataset") != contracts[0]["dataset"]
        or int(receipt.get("calib_seed", -1)) != contracts[0]["calib_seed"]
        or int(receipt.get("global_samples", -1)) != global_samples
        or int(receipt.get("seqlen", -1)) != seqlen
        or int(receipt.get("partition_count", -1)) != len(contracts)
        or int(receipt.get("body_global_ce_count", -1)) != count
        or float(receipt.get("body_global_ce_sum", float("nan"))) != total
        or float(receipt.get("body_global_ce_mean", float("nan"))) != mean
        or count != global_samples * (seqlen - 1)
        or not np.isfinite(total)
        or total <= 0.0
        or any(
            row["execution_identity_sha256"]
            != receipt["execution_identity_sha256"]
            for row in ordered
        )
    ):
        raise SampleParallelProbeError(
            "global importance receipt CE/calibration invariant differs"
        )
    if partition_contract is not None:
        expected_partition = _normalize_partition_contract(
            partition_contract, where="expected global CE partition"
        )
        by_index = {
            int(contract["partition_index"]): contract for contract in contracts
        }
        if by_index.get(int(expected_partition["partition_index"])) != expected_partition:
            raise SampleParallelProbeError(
                "global importance receipt does not contain this exact partition"
            )
    if (
        execution_identity_sha256 is not None
        and receipt["execution_identity_sha256"]
        != str(execution_identity_sha256)
    ):
        raise SampleParallelProbeError(
            "global importance receipt execution identity differs"
        )
    return {**receipt, "receipt_sha256": str(digest)}


def load_global_importance_receipt(
    path: str | Path,
    *,
    partition_contract: Mapping[str, object],
    execution_identity_sha256: str,
) -> dict[str, object]:
    try:
        receipt = _strict_json_loads(
            Path(path).read_text(encoding="utf-8"),
            where="global importance receipt",
        )
    except SampleParallelProbeError:
        raise
    except (OSError, ValueError) as exc:
        raise SampleParallelProbeError(
            f"global importance receipt is unreadable: {path}"
        ) from exc
    if not isinstance(receipt, Mapping):
        raise SampleParallelProbeError("global importance receipt is malformed")
    return validate_global_importance_receipt(
        receipt,
        partition_contract=partition_contract,
        execution_identity_sha256=execution_identity_sha256,
    )


def global_activation_row_identity(
    local_rows: torch.Tensor, partition_contract: Mapping[str, object],
) -> torch.Tensor:
    """Map local flattened sample/token rows onto the immutable global grid."""
    partition = _normalize_partition_contract(
        partition_contract, where="activation row partition"
    )
    if not isinstance(local_rows, torch.Tensor) or local_rows.ndim != 1:
        raise SampleParallelProbeError(
            "activation rows must be a one-dimensional tensor"
        )
    # Partition contracts require one contiguous sample range, so flattened
    # sample/token identities differ by one constant.  Keeping the addition on
    # ``local_rows.device`` is the no-D2H hot-path contract.
    offset = int(partition["sample_start"]) * int(partition["seqlen"])
    return local_rows.detach().to(dtype=torch.long) + offset


def _device_priority_topk_indices(
    priorities: torch.Tensor,
    rows_limit: int,
) -> torch.Tensor:
    """Return canonical smallest-priority indices without a host read.

    The v2 PRP is a permutation over the closed uint32 row domain, so valid
    distinct rows have distinct priorities.  CUDA ``topk`` therefore has one
    mathematically unique result independent of tie/stability behavior.
    """
    if priorities.ndim != 1 or priorities.dtype != torch.long:
        raise SampleParallelProbeError(
            "activation priorities must be one-dimensional int64"
        )
    limit = int(rows_limit)
    if limit < 1 or priorities.numel() < 1:
        raise SampleParallelProbeError(
            "activation top-k needs positive rows and limit"
        )
    keep = min(limit, int(priorities.numel()))
    _values, indices = torch.topk(
        priorities, keep, largest=False, sorted=True,
    )
    return indices


class ActivationPriorityPlanCache:
    """Layer-scoped GPU priority/selection plans shared by fused siblings.

    The cache owns only deterministic row-selection tensors.  It is not an
    activation or weight cache and is cleared at the existing per-layer
    activation flush.  One full priority vector and top-R selection are shared
    by q/k/v or gate/up siblings through ``activation_priority_group``.
    """

    def __init__(self, partition_contract: Mapping[str, object]):
        self.partition = _normalize_partition_contract(
            partition_contract, where="activation priority plan partition"
        )
        self.global_rows = validate_activation_priority_domain(
            int(self.partition["global_samples"]),
            int(self.partition["seqlen"]),
        )
        self.candidate_rows = int(self.partition["local_samples"]) * int(
            self.partition["seqlen"]
        )
        self.global_offset = int(self.partition["sample_start"]) * int(
            self.partition["seqlen"]
        )
        if self.global_offset + self.candidate_rows > self.global_rows:
            raise SampleParallelProbeError(
                "activation priority plan exceeds the global row domain"
            )
        self._priorities: dict[tuple[str, str], torch.Tensor] = {}
        self._top_rows: dict[
            tuple[str, str, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @staticmethod
    def _device_key(device: torch.device) -> str:
        return str(device)

    def full_priorities(
        self, qname: str, *, device: torch.device,
    ) -> torch.Tensor:
        group = activation_priority_group(qname)
        key = (self._device_key(device), group)
        cached = self._priorities.get(key)
        if cached is None:
            rows = torch.arange(
                self.candidate_rows, device=device, dtype=torch.long,
            )
            global_rows = rows + self.global_offset
            cached = activation_row_priorities(
                str(self.partition["global_calibration_hash"]),
                qname,
                global_rows,
            )
            self._priorities[key] = cached
        return cached

    def lookup(
        self, qname: str, local_rows: torch.Tensor,
    ) -> torch.Tensor:
        rows = local_rows.detach().to(dtype=torch.long)
        priorities = self.full_priorities(qname, device=rows.device)
        return priorities.index_select(0, rows)

    def top_rows(
        self,
        qname: str,
        *,
        device: torch.device,
        rows_limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group = activation_priority_group(qname)
        limit = int(rows_limit)
        key = (self._device_key(device), group, limit)
        cached = self._top_rows.get(key)
        if cached is None:
            priorities = self.full_priorities(qname, device=device)
            indices = _device_priority_topk_indices(priorities, limit)
            cached = (
                indices,
                priorities.index_select(0, indices),
            )
            self._top_rows[key] = cached
        return cached

    def clear(self) -> None:
        self._priorities.clear()
        self._top_rows.clear()


def select_activation_rows_by_global_priority(
    inputs: torch.Tensor,
    local_rows: torch.Tensor,
    *,
    qname: str,
    partition_contract: Mapping[str, object],
    rows_limit: int,
    priority_plan_cache: ActivationPriorityPlanCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select local rows by their globally stable priority.

    The returned row indices stay shard-local.  The strict merger verifies
    their priorities after adding ``sample_start * seqlen`` and publishes
    global row indices in the merged ordinary cache.
    """
    limit = int(rows_limit)
    if limit < 1:
        raise SampleParallelProbeError("activation rows_limit must be positive")
    rows = local_rows.detach().to(
        device=inputs.device, dtype=torch.long,
    ).reshape(-1)
    if int(inputs.shape[0]) != int(rows.numel()):
        raise SampleParallelProbeError("activation inputs and local rows differ")
    if priority_plan_cache is not None:
        priorities = priority_plan_cache.lookup(qname, rows)
    else:
        global_rows = global_activation_row_identity(rows, partition_contract)
        priorities = activation_row_priorities(
            str(partition_contract["global_calibration_hash"]),
            qname,
            global_rows,
        )
    index = _device_priority_topk_indices(priorities, limit)
    return (
        inputs.index_select(0, index),
        rows.index_select(0, index),
        priorities.index_select(0, index),
    )


def merge_activation_priority_reservoir(
    *,
    prior_inputs: torch.Tensor | None,
    prior_local_rows: torch.Tensor | None,
    prior_priorities: torch.Tensor | None,
    new_inputs: torch.Tensor,
    new_local_rows: torch.Tensor,
    qname: str,
    partition_contract: Mapping[str, object],
    rows_limit: int,
    priority_plan_cache: ActivationPriorityPlanCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retain exact local top-R candidates across repeated hook calls."""
    new_rows = new_local_rows.detach().to(
        device=new_inputs.device, dtype=torch.long,
    )
    if int(new_inputs.shape[0]) != int(new_rows.numel()):
        raise SampleParallelProbeError("activation inputs and local rows differ")
    if priority_plan_cache is not None:
        new_priorities = priority_plan_cache.lookup(qname, new_rows)
    else:
        new_global = global_activation_row_identity(
            new_rows, partition_contract
        )
        new_priorities = activation_row_priorities(
            str(partition_contract["global_calibration_hash"]),
            qname,
            new_global,
        )
    if prior_inputs is None:
        combined_inputs = new_inputs
        combined_local = new_rows
        combined_priorities = new_priorities
    else:
        if prior_local_rows is None or prior_priorities is None:
            raise SampleParallelProbeError("activation priority reservoir is incomplete")
        prior_rows = prior_local_rows.detach().to(
            device=new_inputs.device, dtype=torch.long,
        )
        retained_priorities = prior_priorities.detach().to(
            device=new_inputs.device, dtype=torch.long,
        )
        if (
            int(prior_inputs.shape[0]) != int(prior_rows.numel())
            or int(prior_rows.numel()) != int(retained_priorities.numel())
        ):
            raise SampleParallelProbeError(
                "activation priority reservoir cardinalities differ"
            )
        combined_inputs = torch.cat([prior_inputs, new_inputs], dim=0)
        combined_local = torch.cat([
            prior_rows, new_rows,
        ], dim=0)
        combined_priorities = torch.cat([
            retained_priorities, new_priorities,
        ], dim=0)
    index = _device_priority_topk_indices(
        combined_priorities, int(rows_limit)
    )
    return (
        combined_inputs.index_select(0, index),
        combined_local.index_select(0, index),
        combined_priorities.index_select(0, index),
    )


def activation_cache_shard_stamp(
    partition_contract: Mapping[str, object],
    *,
    qname: str,
    rows_limit: int,
    candidate_rows: int,
    execution_identity_sha256: str,
) -> dict[str, object]:
    limit = int(rows_limit)
    candidates = int(candidate_rows)
    if limit < 1 or candidates < 1:
        raise SampleParallelProbeError(
            "activation-cache priority cardinalities must be positive"
        )
    execution_digest = str(execution_identity_sha256)
    if re.fullmatch(r"[0-9a-f]{64}", execution_digest) is None:
        raise SampleParallelProbeError(
            "activation-cache execution identity must be a SHA-256 digest"
        )
    return {
        "schema": ACTIVATION_CACHE_SHARD_SCHEMA,
        "priority_schema": ACTIVATION_PRIORITY_SCHEMA,
        "priority_group": activation_priority_group(qname),
        "selection": "local_top_r",
        "row_indices_scope": ROW_INDICES_SCOPE,
        "layout": "dense_linear",
        "global_calibration_hash": str(
            partition_contract["global_calibration_hash"]
        ),
        "execution_identity_sha256": execution_digest,
        "partition_index": int(partition_contract["partition_index"]),
        "rows_limit": limit,
        "candidate_rows": candidates,
    }


def activation_scope_receipt() -> dict[str, object]:
    """Declare the exact v1 cache universe independently of probe qnames."""
    return {
        "schema": ACTIVATION_SCOPE_SCHEMA,
        "included_layout": "dense_body_linear",
        "row_indices_scope": ROW_INDICES_SCOPE,
        "selection": "local_top_r_then_global_top_r",
        "priority_schema": ACTIVATION_PRIORITY_SCHEMA,
        "priority_key": (
            "blake2b64(domain_v2||calibration_hash_bytes||"
            "le_u32(group_utf8_len)||group_utf8)_le_u32_k0_k1"
        ),
        "priority_function": "fmix32(fmix32(row_xor_k0)_xor_k1)",
        "priority_domain": "global_flat_sample_token_row_uint32",
        "priority_domain_max_rows": ACTIVATION_PRIORITY_MAX_ROWS,
        "priority_uniqueness": "fmix32x2_permutation",
        "priority_device_residency": "hook_until_layer_flush",
        "cache_dtype": "torch.float32",
        "resident_lm_head_cache": "omitted_terminal_bf16",
        "mtp_cache": "omitted_no_direct_sample_token_row_map_v1",
        "routed_expert_cache": "unsupported_fail_closed",
        "packed_expert_cache": "unsupported_fail_closed",
        "h_detail": "unsupported_fail_closed",
        "separate_probe_and_activation_qname_manifests_required": True,
    }


def _manifest_with_digest(
    *, kind: str, source_census_sha256: str,
    entries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": QNAME_MANIFEST_SCHEMA,
        "kind": str(kind),
        "source_census_sha256": str(source_census_sha256),
        "entries": {
            str(name): dict(entry) for name, entry in sorted(entries.items())
        },
    }
    return {
        **body,
        "identity_sha256": _canonical_sha256(
            body, where=f"{kind} qname manifest"
        ),
    }


# The RTX4090 Qwen3.8 source qname census lived here: it derived the
# sample-parallel qname universe from an authoritative source layout that
# only the Gridbook lane's producer shipped.  That lane retired
# 2026-09-02 (see archive/gridbook_lane_2026-09-02/), taking
# prismaquant.rtx4090_artifact_census and prismaquant.rtx4090_qwen38_policy
# with it, so the builder, its two private helpers, the worker-local
# re-attestation that called it, and the run-contract builder are gone.
# What survives validates run contracts and censuses already on disk,
# which is all the merge side needs.


def _validate_qname_manifest(
    raw: Mapping[str, object], *, kind: str, source_digest: str,
) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema", "kind", "source_census_sha256", "entries",
        "identity_sha256",
    }:
        raise SampleParallelProbeError(f"{kind} qname manifest fields differ")
    body = dict(raw)
    digest = body.pop("identity_sha256", None)
    if (
        body.get("schema") != QNAME_MANIFEST_SCHEMA
        or body.get("kind") != kind
        or body.get("source_census_sha256") != source_digest
        or digest != _canonical_sha256(body, where=f"{kind} qname manifest")
    ):
        raise SampleParallelProbeError(f"{kind} qname manifest identity differs")
    entries = body.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise SampleParallelProbeError(f"{kind} qname manifest is empty")
    normalized_entries: dict[str, dict[str, object]] = {}
    entry_keys = {
        "source_tensor", "source_dtype", "shape", "disposition",
        "token_rows_per_sample", "terminal_format",
    }
    for raw_name, raw_entry in entries.items():
        name = str(raw_name)
        if not name or not isinstance(raw_entry, Mapping) or set(raw_entry) != entry_keys:
            raise SampleParallelProbeError(
                f"{kind} qname manifest entry {name!r} differs"
            )
        entry = dict(raw_entry)
        if (
            not str(entry["source_tensor"])
            or str(entry["source_dtype"]) != "BF16"
            or not isinstance(entry["shape"], Sequence)
            or isinstance(entry["shape"], (str, bytes))
            or len(entry["shape"]) != 2
            or any(type(value) is not int for value in entry["shape"])
            or any(int(value) < 1 for value in entry["shape"])
        ):
            raise SampleParallelProbeError(
                f"{kind} qname manifest entry {name!r} source shape differs"
            )
        normalized_entries[name] = {
            **entry,
            "source_tensor": str(entry["source_tensor"]),
            "source_dtype": str(entry["source_dtype"]),
            "shape": [int(value) for value in entry["shape"]],
        }
    return {**body, "entries": dict(sorted(normalized_entries.items())),
            "identity_sha256": str(digest)}


def validate_qname_census(raw: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "source_census", "probe_qname_manifest",
        "activation_qname_manifest", "terminal_qname_manifest",
        "identity_sha256",
    }:
        raise SampleParallelProbeError("qname census bundle fields differ")
    source_raw = raw.get("source_census")
    source_keys = {
        "schema", "model", "producer_profile_schema", "producer_profile_id",
        "source_layout", "source_config_sha256",
        "source_tensor_manifest_sha256", "source_weight_map_sha256",
        "source_tensor_count", "source_linear_count", "linear_entries",
        "source_model_identity",
        "identity_sha256",
    }
    if not isinstance(source_raw, Mapping) or set(source_raw) != source_keys:
        raise SampleParallelProbeError("source qname census fields differ")
    source_body = dict(source_raw)
    source_digest = source_body.pop("identity_sha256", None)
    # The producer-profile equality clauses checked this census against the
    # archived RTX4090 policy identity; that policy retired with the
    # Gridbook lane on 2026-09-02 (archive/gridbook_lane_2026-09-02/).
    # The key-set check above still requires the two recorded keys, because
    # every run contract on disk carries them.
    if (
        source_body.get("schema") != SOURCE_QNAME_CENSUS_SCHEMA
        or not str(source_body.get("model", ""))
        or source_body.get("source_layout") not in {
            "flattened_text", "official_wrapper"
        }
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(source_body.get(key, "")))
            for key in (
                "source_config_sha256", "source_tensor_manifest_sha256",
                "source_weight_map_sha256",
            )
        )
        or type(source_body.get("source_tensor_count")) is not int
        or type(source_body.get("source_linear_count")) is not int
        or not re.fullmatch(r"[0-9a-f]{64}", str(source_digest or ""))
        or source_digest != _canonical_sha256(
            source_body, where="sample-parallel source qname census"
        )
    ):
        raise SampleParallelProbeError("source qname census identity differs")
    linear_entries = source_body.get("linear_entries")
    if not isinstance(linear_entries, Mapping) or int(
        source_body.get("source_linear_count", -1)
    ) != len(linear_entries) or int(
        source_body.get("source_tensor_count", -1)
    ) < len(linear_entries):
        raise SampleParallelProbeError("source qname census Linear cover differs")
    model_content = source_body.get("source_model_identity")
    model_content_keys = {
        "schema", "derivation", "upstream_content_sha256",
        "upstream_portable_content_sha256", "content_sha256",
        "resolved_commit", "checkpoint_tensors", "checkpoint_shards",
        "checkpoint_weight_map_sha256", "shards", "identity_sha256",
    }
    if not isinstance(model_content, Mapping) or set(model_content) != model_content_keys:
        raise SampleParallelProbeError("source model-content identity fields differ")
    model_content_body = dict(model_content)
    model_content_digest = model_content_body.pop("identity_sha256", None)
    model_shards = model_content_body.get("shards")
    upstream_content = model_content_body.get("upstream_content_sha256")
    upstream_portable_content = model_content_body.get(
        "upstream_portable_content_sha256"
    )
    if (
        model_content_body.get("schema")
        != SOURCE_MODEL_CONTENT_SCHEMA
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(model_content_body.get("content_sha256", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(model_content_body.get("checkpoint_weight_map_sha256", "")),
        )
        or not isinstance(model_shards, Sequence)
        or isinstance(model_shards, (str, bytes))
        or len(model_shards) != int(model_content_body.get("checkpoint_shards", -1))
        or type(model_content_body.get("checkpoint_shards")) is not int
        or type(model_content_body.get("checkpoint_tensors")) is not int
        or int(model_content_body.get("checkpoint_tensors", -1)) < 1
        or model_content_body.get("checkpoint_weight_map_sha256")
        != source_body.get("source_weight_map_sha256")
        or model_content_body.get("content_sha256") != _canonical_sha256(
            {
                "source_config_sha256": source_body.get(
                    "source_config_sha256"
                ),
                "checkpoint_weight_map_sha256": model_content_body.get(
                    "checkpoint_weight_map_sha256"
                ),
                "shards": model_shards,
            },
            where="sample-parallel model content",
        )
        or model_content_digest != _canonical_sha256(
            model_content_body, where="sample-parallel model-content identity"
        )
        or (
            upstream_content is not None
            and (
                type(upstream_content) is not str
                or re.fullmatch(r"[0-9a-f]{64}", upstream_content) is None
            )
        )
        or (
            upstream_portable_content is not None
            and (
                type(upstream_portable_content) is not str
                or re.fullmatch(
                    r"[0-9a-f]{64}", upstream_portable_content
                ) is None
            )
        )
        or (upstream_content is None) != (upstream_portable_content is None)
        or (
            model_content_body.get("derivation")
            == "validated_streamed_model_identity_cache_v1"
            and (
                upstream_content is None
                or upstream_portable_content is None
            )
        )
    ):
        raise SampleParallelProbeError("source model-content identity differs")
    for shard in model_shards:
        if (
            not isinstance(shard, Mapping)
            or set(shard) != {"name", "size", "sha256"}
            or type(shard.get("name")) is not str
            or type(shard.get("size")) is not int
            or type(shard.get("sha256")) is not str
            or Path(str(shard.get("name", ""))).name != shard.get("name")
            or int(shard.get("size", -1)) < 1
            or not re.fullmatch(r"[0-9a-f]{64}", str(shard.get("sha256", "")))
        ):
            raise SampleParallelProbeError(
                "source model-content shard identity differs"
            )
    # Reuse the entry validator through a temporary manifest; its digest is
    # derived here, so the source census remains the sole authority.
    all_manifest = _manifest_with_digest(
        kind="source_linear_census", source_census_sha256=str(source_digest),
        entries=linear_entries,  # type: ignore[arg-type]
    )
    all_entries = _validate_qname_manifest(
        all_manifest, kind="source_linear_census",
        source_digest=str(source_digest),
    )["entries"]
    allowed = {
        BODY_STATS_AND_ACTIVATION: (BODY_TOKEN_ROWS, None),
        LM_HEAD_STATS_ONLY: (LM_HEAD_TOKEN_ROWS, "BF16"),
        MTP_STATS_ONLY: (MTP_TOKEN_ROWS, "BF16"),
        TEXT_EXCLUDED: (None, None),
    }
    for name, entry in all_entries.items():
        disposition = entry["disposition"]
        expected = allowed.get(disposition)
        if expected is None or (
            entry["token_rows_per_sample"], entry["terminal_format"]
        ) != expected:
            raise SampleParallelProbeError(
                f"source qname {name!r} disposition contract differs"
            )
        name_disposition = (
            LM_HEAD_STATS_ONLY if name == "lm_head"
            else MTP_STATS_ONLY if name.startswith("mtp.")
            else TEXT_EXCLUDED if name.startswith("model.visual.")
            else BODY_STATS_AND_ACTIVATION
            if name.startswith("model.layers.") else None
        )
        if disposition != name_disposition:
            raise SampleParallelProbeError(
                f"source qname {name!r} namespace/disposition differs"
            )
    probe = _validate_qname_manifest(
        raw["probe_qname_manifest"], kind="full_probe",
        source_digest=str(source_digest),  # type: ignore[arg-type]
    )
    activation = _validate_qname_manifest(
        raw["activation_qname_manifest"], kind="dense_body_activation",
        source_digest=str(source_digest),  # type: ignore[arg-type]
    )
    terminal = _validate_qname_manifest(
        raw["terminal_qname_manifest"], kind="terminal_bf16_stats_only",
        source_digest=str(source_digest),  # type: ignore[arg-type]
    )
    expected_probe = {
        name: entry for name, entry in all_entries.items()
        if entry["disposition"] != TEXT_EXCLUDED
    }
    expected_activation = {
        name: entry for name, entry in all_entries.items()
        if entry["disposition"] == BODY_STATS_AND_ACTIVATION
    }
    expected_terminal = {
        name: entry for name, entry in all_entries.items()
        if entry["disposition"] in {LM_HEAD_STATS_ONLY, MTP_STATS_ONLY}
    }
    if probe["entries"] != expected_probe:
        raise SampleParallelProbeError(
            "full-probe qname manifest is not derived from the source census"
        )
    if activation["entries"] != expected_activation:
        raise SampleParallelProbeError(
            "activation qname manifest is not the dense-body source subset"
        )
    if terminal["entries"] != expected_terminal:
        raise SampleParallelProbeError(
            "terminal qname manifest is not the complete BF16 source subset"
        )
    if set(expected_terminal) != {
        name for name in expected_probe
        if name == "lm_head" or name.startswith("mtp.")
    } or "lm_head" not in expected_terminal or not any(
        name.startswith("mtp.") for name in expected_terminal
    ):
        raise SampleParallelProbeError("terminal BF16/MTP census is incomplete")
    source = {**source_body, "linear_entries": all_entries,
              "identity_sha256": str(source_digest)}
    body = {
        "source_census": source,
        "probe_qname_manifest": probe,
        "activation_qname_manifest": activation,
        "terminal_qname_manifest": terminal,
    }
    bundle_digest = raw.get("identity_sha256")
    if bundle_digest != _canonical_sha256(
        body, where="sample-parallel qname census bundle"
    ):
        raise SampleParallelProbeError("qname census bundle digest differs")
    return {**body, "identity_sha256": str(bundle_digest)}


def stable_source_census_projection(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Project a census onto values that must agree across worker hosts.

    A streamed-model identity cache deliberately contains host-local paths,
    stat fingerprints, and a raw upstream receipt.  The source census also
    records the local model argument.  Those values prove local execution
    provenance but are not cross-host equality fields.  The portable upstream
    receipt, semantic config, deterministic content digest, stable shard
    names/sizes/hashes, header/index digests, counts, and exact Linear/qname
    manifests are the cross-host contract.
    """
    census = validate_qname_census(raw)
    source = census["source_census"]
    model_content = source["source_model_identity"]

    def _stable_manifest(name: str) -> dict[str, object]:
        manifest = census[name]
        return {
            "schema": manifest["schema"],
            "kind": manifest["kind"],
            "entries": copy.deepcopy(manifest["entries"]),
        }

    # Do not carry source/model/manifest identity_sha256 values: each of them
    # transitively includes the host-local model-identity provenance named in
    # the docstring.  This projection gets its own digest after removing it.
    body: dict[str, object] = {
        "schema": SOURCE_CENSUS_STABLE_PROJECTION_SCHEMA,
        "producer_profile_schema": source["producer_profile_schema"],
        "producer_profile_id": source["producer_profile_id"],
        "source_layout": source["source_layout"],
        "source_config_sha256": source["source_config_sha256"],
        "source_tensor_manifest_sha256": source[
            "source_tensor_manifest_sha256"
        ],
        "source_weight_map_sha256": source["source_weight_map_sha256"],
        "source_tensor_count": source["source_tensor_count"],
        "source_linear_count": source["source_linear_count"],
        "model_content": {
            "schema": model_content["schema"],
            "upstream_portable_content_sha256": model_content[
                "upstream_portable_content_sha256"
            ],
            "content_sha256": model_content["content_sha256"],
            "checkpoint_tensors": model_content["checkpoint_tensors"],
            "checkpoint_shards": model_content["checkpoint_shards"],
            "checkpoint_weight_map_sha256": model_content[
                "checkpoint_weight_map_sha256"
            ],
            "shards": copy.deepcopy(model_content["shards"]),
        },
        "linear_entries": copy.deepcopy(source["linear_entries"]),
        "probe_qname_manifest": _stable_manifest("probe_qname_manifest"),
        "activation_qname_manifest": _stable_manifest(
            "activation_qname_manifest"
        ),
        "terminal_qname_manifest": _stable_manifest(
            "terminal_qname_manifest"
        ),
    }
    return {
        **body,
        "identity_sha256": _canonical_sha256(
            body, where="sample-parallel stable worker source projection"
        ),
    }


def _runtime_snapshot_entries(root: Path) -> list[dict[str, object]]:
    """Rebuild the tracked snapshot ledger without executing snapshot code."""
    entries: list[dict[str, object]] = []
    manifest_name = ".prismaquant-runtime-snapshot.json"
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False,
    ):
        current = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            path = current / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                entries.append({
                    "path": relative,
                    "type": "symlink",
                    "target": os.readlink(path),
                })
            elif stat.S_ISDIR(info.st_mode):
                kept_directories.append(name)
            else:
                raise SampleParallelProbeError(
                    f"unsupported runtime snapshot entry: {path}"
                )
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(root).as_posix()
            if relative == manifest_name:
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                entries.append({
                    "path": relative,
                    "type": "symlink",
                    "target": os.readlink(path),
                })
            elif stat.S_ISREG(info.st_mode):
                entries.append({
                    "path": relative,
                    "type": "file",
                    "bytes": int(info.st_size),
                    "executable": bool(info.st_mode & stat.S_IXUSR),
                    "sha256": _sha256_file(path),
                })
            else:
                raise SampleParallelProbeError(
                    f"unsupported runtime snapshot entry: {path}"
                )
    return sorted(entries, key=lambda entry: str(entry["path"]))


def _runtime_snapshot_closure_sha256(
    entries: Sequence[Mapping[str, object]],
) -> str:
    encoded = json.dumps(
        list(entries), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_local_producer_snapshot(
    snapshot_root: str | Path,
    *,
    expected_closure_sha256: str | None = None,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
    require_current_module_inside: bool = True,
) -> dict[str, object]:
    """Re-hash the mounted immutable producer snapshot before reuse.

    The ledger is replayed directly by this already-running producer module;
    candidate verifier source inside the snapshot is data, never executable
    authority.  A trusted host launcher must independently verify the snapshot
    before mounting it; this in-process replay closes later mount/runtime
    mutation, rather than pretending self-verifying code is its own trust root.
    Docker image identity is intentionally *not* inferred here: a process in a
    container cannot authoritatively inspect the host-verified immutable
    registry RepoDigest; that value is an external trusted-launcher
    attestation compared elsewhere.
    """
    supplied_root = Path(snapshot_root)
    if supplied_root.is_symlink():
        raise SampleParallelProbeError(
            "immutable producer snapshot root must not be a symlink"
        )
    root = supplied_root.resolve(strict=True)
    if not root.is_dir():
        raise SampleParallelProbeError(
            "immutable producer snapshot root is not a directory"
        )
    if require_current_module_inside:
        try:
            Path(__file__).resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise SampleParallelProbeError(
                "sample-parallel producer is not executing from the declared "
                "immutable runtime snapshot"
            ) from exc
    manifest_path = root / ".prismaquant-runtime-snapshot.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SampleParallelProbeError(
            "immutable producer snapshot manifest must be one regular file"
        )
    try:
        manifest = _strict_json_loads(
            manifest_path.read_text(encoding="utf-8"),
            where="immutable producer snapshot manifest",
        )
    except SampleParallelProbeError:
        raise
    except (OSError, ValueError) as exc:
        raise SampleParallelProbeError(
            "immutable producer snapshot manifest is unreadable"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise SampleParallelProbeError(
            "immutable producer snapshot manifest is malformed"
        )
    expected_manifest_keys = {
        "schema", "commit", "tree", "closure_sha256", "entry_count",
        "entries",
    }
    if set(manifest) != expected_manifest_keys or manifest.get(
        "schema"
    ) != "prismaquant.runtime_source_snapshot.v1":
        raise SampleParallelProbeError(
            "immutable producer snapshot manifest schema is not closed"
        )
    commit = str(manifest.get("commit", ""))
    tree = str(manifest.get("tree", ""))
    closure = str(manifest.get("closure_sha256", ""))
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", tree) is None
        or re.fullmatch(r"[0-9a-f]{64}", closure) is None
    ):
        raise SampleParallelProbeError(
            "immutable producer snapshot identity is malformed"
        )
    if expected_commit is not None and commit != str(expected_commit):
        raise SampleParallelProbeError(
            "local producer snapshot commit differs from the run contract"
        )
    if expected_tree is not None and tree != str(expected_tree):
        raise SampleParallelProbeError(
            "local producer snapshot tree differs from the run contract"
        )
    if (
        expected_closure_sha256 is not None
        and closure != str(expected_closure_sha256)
    ):
        raise SampleParallelProbeError(
            "local producer snapshot closure differs from the run contract"
        )
    manifest_entries = manifest.get("entries")
    entry_count = manifest.get("entry_count")
    if (
        not isinstance(manifest_entries, list)
        or isinstance(entry_count, bool)
        or not isinstance(entry_count, int)
        or entry_count < 1
        or len(manifest_entries) != entry_count
    ):
        raise SampleParallelProbeError(
            "immutable producer snapshot entry ledger is malformed"
        )
    try:
        observed_entries = _runtime_snapshot_entries(root)
        observed_closure = _runtime_snapshot_closure_sha256(observed_entries)
    except Exception as exc:
        raise SampleParallelProbeError(
            f"immutable producer snapshot verification failed: {exc}"
        ) from exc
    if observed_entries != manifest_entries:
        raise SampleParallelProbeError(
            "immutable producer snapshot files differ from the tracked ledger"
        )
    if observed_closure != closure:
        raise SampleParallelProbeError(
            "immutable producer snapshot closure differs from its ledger"
        )
    for directory in (root / "prismaquant", root / "tools"):
        if directory.is_symlink() or not directory.is_dir():
            raise SampleParallelProbeError(
                f"immutable producer runtime directory is unsafe: {directory}"
            )
    for required in (
        root / "prismaquant" / "__init__.py",
        root / "prismaquant" / "sample_parallel_probe.py",
        root / "prismaquant" / "incremental_probe.py",
        root / "tools" / "prismaquant_source_bootstrap.py",
        root / "tools" / "prismaquant_runtime_snapshot.py",
    ):
        if required.is_symlink() or not required.is_file():
            raise SampleParallelProbeError(
                f"immutable producer runtime entry is unsafe: {required}"
            )
    return {
        "commit": commit,
        "tree": tree,
        "closure_sha256": closure,
        "snapshot_root": str(root),
    }


def importance_execution_identity_sha256(
    execution_identity: Mapping[str, object],
    qname_census: Mapping[str, object],
) -> str:
    """Bind CE reuse to source content and the complete producer contract."""
    census = validate_qname_census(qname_census)
    execution = validate_execution_identity(
        execution_identity, qname_census=census
    )
    stable_source = stable_source_census_projection(census)
    body = {
        "schema": IMPORTANCE_EXECUTION_BINDING_SCHEMA,
        "execution_identity_sha256": execution["identity_sha256"],
        "stable_source_projection_sha256": stable_source["identity_sha256"],
        "producer_snapshot_sha256": execution["producer_snapshot_sha256"],
        "producer_snapshot_commit": execution["producer_snapshot_commit"],
        "producer_snapshot_tree": execution["producer_snapshot_tree"],
        "container_image_digest": execution["container_image_digest"],
    }
    return _canonical_sha256(
        body, where="sample-parallel importance execution binding"
    )


def build_execution_identity(
    *,
    model: str,
    dataset: str,
    calib_seed: int,
    dtype: str,
    importance_weighting: bool,
    emit_marginals: bool,
    activation_rows_limit: int,
    qname_census: Mapping[str, object],
    producer_snapshot_sha256: str,
    producer_snapshot_commit: str,
    producer_snapshot_tree: str,
    container_image_digest: str,
) -> dict[str, object]:
    census = validate_qname_census(qname_census)
    if int(activation_rows_limit) != 1024:
        raise SampleParallelProbeError(
            "sample-parallel v1 requires activation_rows_limit=1024"
        )
    if str(census["source_census"]["model"]) != str(model):
        raise SampleParallelProbeError(
            "execution model differs from the source qname census"
        )
    body: dict[str, object] = {
        "schema": EXECUTION_IDENTITY_SCHEMA,
        "model": str(model),
        "dataset": str(dataset),
        "calib_seed": int(calib_seed),
        "dtype": str(dtype),
        "calibration_modality": "text-only",
        "probe_schedule": "unified_full_body_mtp_lm_head_text_only_v1",
        "include_visual": False,
        "importance_weighting": bool(importance_weighting),
        "importance_contract": (
            IMPORTANCE_ENABLED_CONTRACT
            if importance_weighting else IMPORTANCE_DISABLED_CONTRACT
        ),
        "activation_scope": activation_scope_receipt(),
        "activation_rows_limit": int(activation_rows_limit),
        "emit_marginals": bool(emit_marginals),
        "marginal_contract": (
            MARGINAL_ENABLED_CONTRACT if emit_marginals
            else MARGINAL_DISABLED_CONTRACT
        ),
        "estimator_contract": ESTIMATOR_CONTRACT,
        "math_contract": dict(MATH_CONTRACT),
        "h_detail": False,
        "routed_packed_stats": "unsupported_fail_closed_v1",
        "producer_snapshot_sha256": str(producer_snapshot_sha256),
        "producer_snapshot_commit": str(producer_snapshot_commit),
        "producer_snapshot_tree": str(producer_snapshot_tree),
        "container_image_digest": str(container_image_digest),
        "source_census_sha256": census["source_census"]["identity_sha256"],
        "model_content_sha256": census["source_census"][
            "source_model_identity"
        ]["content_sha256"],
        "probe_qname_manifest_sha256": census[
            "probe_qname_manifest"
        ]["identity_sha256"],
        "activation_qname_manifest_sha256": census[
            "activation_qname_manifest"
        ]["identity_sha256"],
        "terminal_qname_manifest_sha256": census[
            "terminal_qname_manifest"
        ]["identity_sha256"],
    }
    return validate_execution_identity({
        **body,
        "identity_sha256": _canonical_sha256(
            body, where="sample-parallel execution identity"
        ),
    }, qname_census=census)


def validate_execution_identity(
    raw: Mapping[str, object], *, qname_census: Mapping[str, object],
) -> dict[str, object]:
    census = validate_qname_census(qname_census)
    if not isinstance(raw, Mapping) or set(raw) != _EXECUTION_IDENTITY_KEYS:
        raise SampleParallelProbeError(
            "execution identity fields differ from the closed v1 contract"
        )
    body = dict(raw)
    digest = body.pop("identity_sha256", None)
    if (
        body.get("schema") != EXECUTION_IDENTITY_SCHEMA
        or digest != _canonical_sha256(
            body, where="sample-parallel execution identity"
        )
        or not str(body.get("model", ""))
        or not str(body.get("dataset", ""))
        or type(body.get("model")) is not str
        or type(body.get("dataset")) is not str
        or type(body.get("calib_seed")) is not int
        or int(body.get("calib_seed", -1)) < 0
        or body.get("dtype") not in {"bf16", "fp16", "fp32"}
        or body.get("calibration_modality") != "text-only"
        or body.get("probe_schedule")
        != "unified_full_body_mtp_lm_head_text_only_v1"
        or body.get("include_visual") is not False
        or type(body.get("importance_weighting")) is not bool
        or type(body.get("emit_marginals")) is not bool
        or type(body.get("activation_rows_limit")) is not int
        or int(body.get("activation_rows_limit", 0)) != 1024
        or body.get("activation_scope") != activation_scope_receipt()
        or body.get("estimator_contract") != ESTIMATOR_CONTRACT
        or body.get("math_contract") != MATH_CONTRACT
        or body.get("h_detail") is not False
        or body.get("routed_packed_stats")
        != "unsupported_fail_closed_v1"
        or re.fullmatch(
            r"[0-9a-f]{64}", str(body.get("producer_snapshot_sha256", ""))
        ) is None
        or re.fullmatch(
            r"[0-9a-f]{40}", str(body.get("producer_snapshot_commit", ""))
        ) is None
        or re.fullmatch(
            r"[0-9a-f]{40}", str(body.get("producer_snapshot_tree", ""))
        ) is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(body.get("container_image_digest", ""))
        ) is None
    ):
        raise SampleParallelProbeError("execution identity contract differs")
    importance = bool(body["importance_weighting"])
    marginals = bool(body["emit_marginals"])
    if body.get("importance_contract") != (
        IMPORTANCE_ENABLED_CONTRACT if importance else IMPORTANCE_DISABLED_CONTRACT
    ) or body.get("marginal_contract") != (
        MARGINAL_ENABLED_CONTRACT if marginals else MARGINAL_DISABLED_CONTRACT
    ):
        raise SampleParallelProbeError(
            "execution importance/marginal contract differs"
        )
    expected_digests = {
        "source_census_sha256": census["source_census"]["identity_sha256"],
        "model_content_sha256": census["source_census"][
            "source_model_identity"
        ]["content_sha256"],
        "probe_qname_manifest_sha256": census[
            "probe_qname_manifest"
        ]["identity_sha256"],
        "activation_qname_manifest_sha256": census[
            "activation_qname_manifest"
        ]["identity_sha256"],
        "terminal_qname_manifest_sha256": census[
            "terminal_qname_manifest"
        ]["identity_sha256"],
    }
    if any(body.get(key) != value for key, value in expected_digests.items()):
        raise SampleParallelProbeError(
            "execution identity qname/source census digest differs"
        )
    if body.get("model") != census["source_census"]["model"]:
        raise SampleParallelProbeError(
            "execution identity model differs from source census"
        )
    return {**body, "identity_sha256": str(digest)}


def validate_run_contract(raw: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema", "execution_identity", "qname_census", "identity_sha256",
    } or raw.get("schema") != RUN_CONTRACT_SCHEMA:
        raise SampleParallelProbeError("sample-parallel run contract differs")
    census = validate_qname_census(
        raw.get("qname_census")  # type: ignore[arg-type]
    )
    execution = validate_execution_identity(
        raw.get("execution_identity"),  # type: ignore[arg-type]
        qname_census=census,
    )
    body = {
        "schema": RUN_CONTRACT_SCHEMA,
        "execution_identity": execution,
        "qname_census": census,
    }
    if raw.get("identity_sha256") != _canonical_sha256(
        body, where="sample-parallel run contract"
    ):
        raise SampleParallelProbeError("sample-parallel run contract digest differs")
    return {**body, "identity_sha256": str(raw["identity_sha256"])}


def token_rows_per_sample(token_rule: str, seqlen: int) -> int:
    length = int(seqlen)
    if token_rule == BODY_TOKEN_ROWS:
        value = length
    elif token_rule == LM_HEAD_TOKEN_ROWS:
        value = length - 1
    elif token_rule == MTP_TOKEN_ROWS:
        value = length - 2
    else:
        raise SampleParallelProbeError(
            f"unsupported qname token-cover rule {token_rule!r}"
        )
    if value < 1:
        raise SampleParallelProbeError(
            f"seqlen={length} is too short for {token_rule}"
        )
    return value


def _validate_partition_cover(
    contracts: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], str, int, int]:
    if not contracts:
        raise SampleParallelProbeError("no sample partitions to merge")
    normalized = [
        _normalize_partition_contract(
            contract, where=f"sample partition {index}"
        )
        for index, contract in enumerate(contracts)
    ]
    common_fields = (
        "global_calibration_hash", "calibration_artifact_sha256",
        "global_samples", "seqlen", "partition_count", "dataset", "model",
        "calib_seed",
    )
    reference = normalized[0]
    for contract in normalized[1:]:
        for key in common_fields:
            if contract.get(key) != reference.get(key):
                raise SampleParallelProbeError(
                    f"sample partitions disagree at {key}"
                )
    expected_count = int(reference.get("partition_count", -1))
    if expected_count != len(normalized):
        raise SampleParallelProbeError(
            f"sample cover has {len(normalized)} partitions, expected {expected_count}"
        )
    normalized.sort(key=lambda item: int(item.get("partition_index", -1)))
    if [int(item.get("partition_index", -1)) for item in normalized] != list(
        range(expected_count)
    ):
        raise SampleParallelProbeError("sample partition indices are not exact")
    flat: list[int] = []
    cursor = 0
    for item in normalized:
        if int(item["sample_start"]) != cursor:
            raise SampleParallelProbeError(
                "sample partitions are not a contiguous canonical cover"
            )
        flat.extend(int(value) for value in item.get("sample_indices", ()))
        cursor = int(item["sample_stop"])
    if len(flat) != len(set(flat)):
        raise SampleParallelProbeError("sample partitions overlap")
    global_samples = int(reference.get("global_samples", -1))
    if sorted(flat) != list(range(global_samples)):
        raise SampleParallelProbeError("sample partitions do not exactly cover global samples")
    return (
        normalized,
        str(reference["global_calibration_hash"]),
        global_samples,
        int(reference["seqlen"]),
    )


def _load_json_mapping(path: str | Path) -> dict[str, object]:
    try:
        value = _strict_json_loads(
            Path(path).read_text(encoding="utf-8"),
            where=f"JSON artifact {path}",
        )
    except SampleParallelProbeError:
        raise
    except (OSError, ValueError) as exc:
        raise SampleParallelProbeError(f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise SampleParallelProbeError(f"JSON artifact is not a mapping: {path}")
    return dict(value)


def publish_sample_parallel_merge_bundle(
    merged_probe: Mapping[str, object],
    cache_dirs_by_shard: Mapping[int, str | Path],
    output_bundle: str | Path,
    *,
    expected_cover: Mapping[str, object],
    max_rows: int,
) -> dict[str, object]:
    """Atomically publish the probe/cache pair as one no-clobber directory.

    The visible output name is created only after both payloads and a final
    commit manifest are durable inside one same-filesystem staging directory.
    A crash can therefore leave only an unreferenced hidden staging directory,
    never a visible half-pair; an exact retry remains safe.
    """
    import pickle

    from prismaquant.sample_parallel_probe_merge import (
        _publish_directory_no_clobber,
        merge_sample_parallel_activation_caches,
    )

    destination = Path(output_bundle)
    if destination.exists():
        raise SampleParallelProbeError(
            f"merge bundle output already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.sample-merge-bundle-",
        dir=destination.parent,
    ))
    try:
        activation_manifest = merge_sample_parallel_activation_caches(
            cache_dirs_by_shard,
            temporary / MERGE_BUNDLE_ACTIVATIONS,
            expected_cover=expected_cover,
            max_rows=max_rows,
        )
        if os.environ.get(
            "PRISMAQUANT_TEST_FAULT_SAMPLE_MERGE_AFTER_CACHE"
        ) == "1":
            raise SampleParallelProbeError(
                "injected sample merge failure after cache staging"
            )

        probe_bytes = pickle.dumps(
            dict(merged_probe), protocol=pickle.HIGHEST_PROTOCOL
        )
        _atomic_write_bytes_no_clobber(
            temporary / MERGE_BUNDLE_PROBE, probe_bytes
        )

        probe_meta = merged_probe.get("meta")
        probe_merge = (
            probe_meta.get("sample_parallel_merge")
            if isinstance(probe_meta, Mapping) else None
        )
        cover_identity = activation_manifest.get("cover_identity_sha256")
        if (
            not isinstance(probe_merge, Mapping)
            or probe_merge.get("cover_identity_sha256") != cover_identity
        ):
            raise SampleParallelProbeError(
                "probe and activation merge outputs have different covers"
            )
        commit: dict[str, object] = {
            "schema": MERGE_BUNDLE_COMMIT_SCHEMA,
            "cover_identity_sha256": cover_identity,
            "execution_identity_sha256": expected_cover[
                "execution_identity"
            ]["identity_sha256"],
            "probe": {
                "path": MERGE_BUNDLE_PROBE,
                "bytes": len(probe_bytes),
                "sha256": hashlib.sha256(probe_bytes).hexdigest(),
            },
            "activation_cache": {
                "path": MERGE_BUNDLE_ACTIVATIONS,
                "manifest": (
                    f"{MERGE_BUNDLE_ACTIVATIONS}/sample_parallel_merge.json"
                ),
                "manifest_identity_sha256": activation_manifest[
                    "identity_sha256"
                ],
            },
            "complete_probe_activation_pair": True,
            "atomic_directory_publication": True,
        }
        commit["identity_sha256"] = _canonical_sha256(
            commit, where="sample-parallel merge bundle commit"
        )
        _atomic_write_bytes_no_clobber(
            temporary / MERGE_BUNDLE_COMMIT,
            json.dumps(commit, sort_keys=True, indent=2).encode("utf-8")
            + b"\n",
        )
        _fsync_directory(temporary)
        _publish_directory_no_clobber(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return commit


def validate_sample_parallel_merge_bundle(
    bundle_dir: str | Path,
    *,
    expected_cover: Mapping[str, object] | None = None,
    capture_consumables: bool = False,
) -> dict[str, object]:
    """Replay the complete no-clobber probe/cache/commit publication.

    Consumers must enter through this validator instead of validating a probe
    pickle or activation directory independently.  It verifies exact bundle
    topology, the commit checksum and member bytes, the reducer's closed probe
    rows, and every activation payload through the independent merge verifier.
    """
    import pickle

    from prismaquant.sample_parallel_probe_merge import (
        ACTIVATION_CACHE_MANIFEST,
        _validate_cover,
        validate_merged_activation_cache_output,
        validate_sample_parallel_merged_stat_row,
    )

    bundle = Path(bundle_dir)
    if bundle.is_symlink() or not bundle.is_dir():
        raise SampleParallelProbeError(
            "sample-parallel merge bundle is not a regular directory"
        )
    expected_entries = {
        MERGE_BUNDLE_PROBE, MERGE_BUNDLE_ACTIVATIONS, MERGE_BUNDLE_COMMIT,
    }
    entries = {path.name: path for path in bundle.iterdir()}
    if set(entries) != expected_entries or any(
        path.is_symlink() for path in entries.values()
    ) or not entries[MERGE_BUNDLE_PROBE].is_file() or not entries[
        MERGE_BUNDLE_COMMIT
    ].is_file() or not entries[MERGE_BUNDLE_ACTIVATIONS].is_dir():
        raise SampleParallelProbeError(
            "sample-parallel merge bundle topology differs"
        )
    bundle_descriptor = _open_directory_nofollow(
        bundle, where="sample-parallel merge bundle"
    )
    try:
        commit_bytes, _commit_sha256, _commit_opened_stat = (
            _open_regular_bytes_at(
            bundle_descriptor,
            MERGE_BUNDLE_COMMIT,
            where="sample-parallel merge bundle commit",
            )
        )
        probe_bytes, probe_sha256, probe_opened_stat = (
            _open_regular_bytes_at(
            bundle_descriptor,
            MERGE_BUNDLE_PROBE,
            where="sample-parallel merge bundle probe",
            )
        )
    finally:
        os.close(bundle_descriptor)
    try:
        commit = _strict_json_loads(
            commit_bytes.decode("utf-8"),
            where="sample-parallel merge bundle commit",
        )
    except SampleParallelProbeError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SampleParallelProbeError(
            "sample-parallel merge bundle commit is unreadable"
        ) from exc
    commit_keys = {
        "schema", "cover_identity_sha256", "execution_identity_sha256",
        "probe", "activation_cache", "complete_probe_activation_pair",
        "atomic_directory_publication", "identity_sha256",
    }
    if not isinstance(commit, Mapping) or set(commit) != commit_keys:
        raise SampleParallelProbeError(
            "sample-parallel merge bundle commit fields differ"
        )
    commit_body = dict(commit)
    commit_identity = commit_body.pop("identity_sha256", None)
    if (
        commit.get("schema") != MERGE_BUNDLE_COMMIT_SCHEMA
        or not isinstance(commit_identity, str)
        or not re.fullmatch(r"[0-9a-f]{64}", commit_identity)
        or commit_identity != _canonical_sha256(
            commit_body, where="sample-parallel merge bundle commit"
        )
        or commit.get("complete_probe_activation_pair") is not True
        or commit.get("atomic_directory_publication") is not True
    ):
        raise SampleParallelProbeError(
            "sample-parallel merge bundle commit identity differs"
        )
    probe_binding = commit.get("probe")
    activation_binding = commit.get("activation_cache")
    if (
        not isinstance(probe_binding, Mapping)
        or set(probe_binding) != {"path", "bytes", "sha256"}
        or not isinstance(activation_binding, Mapping)
        or set(activation_binding) != {
            "path", "manifest", "manifest_identity_sha256",
        }
    ):
        raise SampleParallelProbeError(
            "sample-parallel merge bundle member bindings differ"
        )
    if (
        probe_binding.get("path") != MERGE_BUNDLE_PROBE
        or type(probe_binding.get("bytes")) is not int
        or probe_binding.get("bytes") != len(probe_bytes)
        or probe_binding.get("sha256") != probe_sha256
        or activation_binding.get("path") != MERGE_BUNDLE_ACTIVATIONS
        or activation_binding.get("manifest")
        != f"{MERGE_BUNDLE_ACTIVATIONS}/{ACTIVATION_CACHE_MANIFEST}"
    ):
        raise SampleParallelProbeError(
            "sample-parallel merge bundle member bytes/paths differ"
        )
    try:
        probe = pickle.loads(probe_bytes)
    except Exception as exc:
        raise SampleParallelProbeError(
            "sample-parallel merge bundle probe is unreadable"
        ) from exc
    try:
        current_probe_stat = _opened_stat_identity(
            entries[MERGE_BUNDLE_PROBE].lstat()
        )
    except OSError as exc:
        raise SampleParallelProbeError(
            "sample-parallel merge bundle probe disappeared during validation"
        ) from exc
    if current_probe_stat != probe_opened_stat:
        raise SampleParallelProbeError(
            "sample-parallel merge bundle probe changed during validation"
        )
    if not isinstance(probe, Mapping) or set(probe) != {
        "stats", "router_counts", "router_totals", "router_active_counts",
        "expert_route_stats", "expert_info", "meta",
    }:
        raise SampleParallelProbeError(
            "sample-parallel merge bundle probe fields differ"
        )
    for name in (
        "router_counts", "router_totals", "router_active_counts",
        "expert_route_stats", "expert_info",
    ):
        if not isinstance(probe[name], Mapping) or probe[name]:
            raise SampleParallelProbeError(
                f"sample-parallel merge bundle probe map {name!r} differs"
            )
    meta = probe.get("meta")
    merge_meta = meta.get("sample_parallel_merge") if isinstance(
        meta, Mapping
    ) else None
    embedded_cover = merge_meta.get("cover") if isinstance(
        merge_meta, Mapping
    ) else None
    cover_identity = commit.get("cover_identity_sha256")
    if (
        not isinstance(embedded_cover, Mapping)
        or not isinstance(cover_identity, str)
        or not re.fullmatch(r"[0-9a-f]{64}", cover_identity)
    ):
        raise SampleParallelProbeError(
            "sample-parallel merge bundle probe lacks its exact cover"
        )
    candidate_cover = {**dict(embedded_cover), "identity_sha256": cover_identity}
    try:
        normalized_cover, _ = _validate_cover(candidate_cover)
        if expected_cover is not None:
            expected_normalized, _ = _validate_cover(expected_cover)
            if normalized_cover != expected_normalized:
                raise SampleParallelProbeError(
                    "sample-parallel merge bundle cover differs from expected"
                )
    except SampleParallelProbeError:
        raise
    except Exception as exc:
        raise SampleParallelProbeError(
            f"sample-parallel merge bundle cover is invalid: {exc}"
        ) from exc
    if (
        merge_meta.get("cover_identity_sha256") != cover_identity
        or normalized_cover.get("execution_identity", {}).get(
            "identity_sha256"
        ) != commit.get("execution_identity_sha256")
    ):
        raise SampleParallelProbeError(
            "sample-parallel merge bundle execution/cover identity differs"
        )
    stats = probe.get("stats")
    probe_entries = normalized_cover["qname_census"][
        "probe_qname_manifest"
    ]["entries"]
    if not isinstance(stats, Mapping) or set(stats) != set(probe_entries):
        raise SampleParallelProbeError(
            "sample-parallel merge bundle probe qname cover differs"
        )
    total_samples = int(normalized_cover["total_samples"])
    seqlen = int(normalized_cover["seqlen"])
    require_marginals = bool(
        normalized_cover["execution_identity"]["emit_marginals"]
    )
    try:
        for qname, entry in probe_entries.items():
            expected_tokens = total_samples * token_rows_per_sample(
                str(entry["token_rows_per_sample"]), seqlen
            )
            validate_sample_parallel_merged_stat_row(
                stats[qname], qname=qname,
                expected_tokens=expected_tokens,
                normalization_tokens=total_samples * seqlen,
                expected_shape=tuple(int(value) for value in entry["shape"]),
                require_marginals=require_marginals,
            )
        manifest = validate_merged_activation_cache_output(
            entries[MERGE_BUNDLE_ACTIVATIONS],
            expected_cover=candidate_cover,
        )
    except Exception as exc:
        raise SampleParallelProbeError(
            f"sample-parallel merge bundle member validation failed: {exc}"
        ) from exc
    if activation_binding.get(
        "manifest_identity_sha256"
    ) != manifest.get("identity_sha256"):
        raise SampleParallelProbeError(
            "sample-parallel merge bundle activation manifest differs"
        )
    source_census = normalized_cover["qname_census"]["source_census"]
    source_model = source_census["source_model_identity"]
    source_content = source_model.get("content_sha256")
    source_upstream_content = source_model.get("upstream_content_sha256")
    source_upstream_portable_content = source_model.get(
        "upstream_portable_content_sha256"
    )
    source_census_identity = source_census.get("identity_sha256")
    if (
        not isinstance(source_content, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_content) is None
        or not isinstance(source_census_identity, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_census_identity) is None
    ):
        raise SampleParallelProbeError(
            "sample-parallel merge bundle source identity differs"
        )
    result: dict[str, object] = {
        "schema": MERGE_BUNDLE_COMMIT_SCHEMA,
        "identity_sha256": commit_identity,
        "cover_identity_sha256": cover_identity,
        "execution_identity_sha256": commit["execution_identity_sha256"],
        "probe_sha256": probe_binding["sha256"],
        "probe_bytes": probe_binding["bytes"],
        "activation_manifest_identity_sha256": manifest["identity_sha256"],
        "source_model_content_sha256": source_content,
        "source_model_upstream_content_sha256": source_upstream_content,
        "source_model_upstream_portable_content_sha256": (
            source_upstream_portable_content
        ),
        "source_census_identity_sha256": source_census_identity,
    }
    if capture_consumables:
        # These are the exact in-memory objects validated above.  Strict burn
        # consumers use them instead of reopening pathnames after validation.
        result["_validated_probe_bytes"] = probe_bytes
        result["_validated_probe_payload"] = probe
        result["_validated_activation_manifest"] = manifest
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-calibration")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--nsamples", required=True, type=int)
    prepare.add_argument("--seqlen", required=True, type=int)
    prepare.add_argument("--calib-seed", default=42, type=int)
    prepare.add_argument("--partitions", required=True, type=int)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--manifest-output", required=True)

    source_cache = commands.add_parser("prepare-worker-source-cache")
    source_cache.add_argument("--model", required=True)
    source_cache.add_argument("--output", required=True)
    source_cache.add_argument("--offload-folder", required=True)

    importance = commands.add_parser("merge-importance")
    importance.add_argument("--local-stats", nargs="+", required=True)
    importance.add_argument("--output", required=True)

    cover = commands.add_parser("build-cover")
    cover.add_argument("--calibration-manifest", required=True)
    cover.add_argument("--run-contract", required=True)
    cover.add_argument("--output", required=True)

    merge = commands.add_parser("merge")
    merge.add_argument("--cover", required=True)
    merge.add_argument("--probe-shards", nargs="+", required=True)
    merge.add_argument(
        "--activation-cache", nargs="+", required=True, metavar="INDEX=DIR",
    )
    merge.add_argument(
        "--output-bundle", required=True,
        help=(
            "New atomic directory containing fixed probe.pkl, "
            "activation_cache/, and commit.json outputs."
        ),
    )
    merge.add_argument("--max-rows", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare-worker-source-cache":
        receipt = prepare_worker_source_cache(
            model=args.model,
            output=args.output,
            offload_folder=args.offload_folder,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "prepare-calibration":
        manifest = prepare_global_calibration(
            model=args.model, dataset=args.dataset, nsamples=args.nsamples,
            seqlen=args.seqlen, calib_seed=args.calib_seed,
            partition_count=args.partitions, output=args.output,
            manifest_output=args.manifest_output,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "merge-importance":
        receipt = merge_importance_stats(args.local_stats, args.output)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "build-cover":
        from prismaquant.sample_parallel_probe_merge import (
            build_sample_parallel_cover,
        )
        manifest = validate_calibration_manifest(
            _load_json_mapping(args.calibration_manifest)
        )
        contracts = manifest.get("partition_contracts")
        if not isinstance(contracts, Sequence):
            raise SampleParallelProbeError(
                "calibration manifest lacks partition_contracts"
            )
        run_contract = validate_run_contract(
            _load_json_mapping(args.run_contract)
        )
        cover = build_sample_parallel_cover(
            contracts,
            execution_identity=run_contract["execution_identity"],
            qname_census=run_contract["qname_census"],
        )
        _atomic_write_bytes_no_clobber(
            Path(args.output),
            json.dumps(cover, sort_keys=True, indent=2).encode("utf-8") + b"\n",
        )
        print(args.output)
        return 0

    import pickle
    from prismaquant.sample_parallel_probe_merge import (
        merge_sample_parallel_probe_payloads,
    )

    cover = _load_json_mapping(args.cover)
    if Path(args.output_bundle).exists():
        raise SampleParallelProbeError(
            f"merge bundle output already exists: {args.output_bundle}"
        )
    payloads = []
    for path in args.probe_shards:
        with Path(path).open("rb") as handle:
            payloads.append(pickle.load(handle))
    merged = merge_sample_parallel_probe_payloads(
        payloads, expected_cover=cover,
    )
    cache_dirs: dict[int, str] = {}
    for value in args.activation_cache:
        raw_index, separator, raw_path = value.partition("=")
        if not separator or not raw_path:
            raise SampleParallelProbeError(
                "--activation-cache values must be INDEX=DIR"
            )
        index = int(raw_index)
        if index in cache_dirs:
            raise SampleParallelProbeError("duplicate activation-cache index")
        cache_dirs[index] = raw_path
    publish_sample_parallel_merge_bundle(
        merged, cache_dirs, args.output_bundle,
        expected_cover=cover,
        max_rows=args.max_rows,
    )
    print(args.output_bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
