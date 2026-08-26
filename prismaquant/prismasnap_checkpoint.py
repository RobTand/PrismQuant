"""Strict planning and streaming checkpoint materialization for PrismaSnap.

The runtime state machine is intentionally boring and agent-free:

``PREPARED -> PLANNED -> MATERIALIZED -> VERIFIED -> COMMITTED``.

Every transition is content-bound.  Outputs are written to a sibling
``.prismasnap-incomplete`` directory, each shard is atomically published with
a digest receipt, and the requested output path appears only after the full
tensor census and provenance checks pass.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from .cost_stage_checkpoint import canonical_json_sha256
from .cost_streaming import (
    canonical_streamed_model_semantic_config,
    portable_streamed_model_content_identity,
    validate_streamed_model_identity,
)
from .model_profiles import detect_profile
from .prismasnap import (
    PRISMASNAP_ALGORITHM,
    PrismaSnapConsumer,
    PrismaSnapSearchConfig,
    apply_diagonal_transform,
    search_diagonal_scale,
)


PLAN_SCHEMA = "prismaquant.prismasnap.plan.v1"
PLAN_SET_SCHEMA = "prismaquant.prismasnap.plan_set.v1"
TENSOR_METADATA_SCHEMA = "prismaquant.prismasnap.tensor_metadata.v1"
TENSOR_METADATA_MANIFEST_SCHEMA = (
    "prismaquant.prismasnap.tensor_metadata_manifest.v1"
)
LEGACY_TEXT_PROBE_BINDING_SCHEMA = (
    "prismaquant.prismasnap.legacy_text_probe_binding.v1"
)
PROVENANCE_SCHEMA = "prismaquant.prismasnap.provenance.v1"
SHARD_RECEIPT_SCHEMA = "prismaquant.prismasnap.materialized_shard.v1"
PART_SCHEMA = "prismaquant.prismasnap.checkpoint_part.v1"
PLAN_MERGE_STATE_SCHEMA = "prismaquant.prismasnap.plan_merge_state.v1"
PART_MERGE_STATE_SCHEMA = "prismaquant.prismasnap.part_merge_state.v1"
COLLATED_SHARD_RECEIPT_SCHEMA = (
    "prismaquant.prismasnap.collated_shard_receipt.v1"
)
PLAN_JSON = "plan.json"
PLAN_SCALES = "scales.safetensors"
PROVENANCE_JSON = "prismasnap_provenance.json"
PLAN_MERGE_STATE_JSON = "plan_merge_state.json"
PART_MERGE_STATE_JSON = "part_merge_state.json"
PART_MERGE_RECEIPTS_DIR = ".prismasnap-collation-receipts"
PROBE_BINDING_SUFFIX = ".prismasnap-binding.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_BODY_LAYER = re.compile(r"^model[.]layers[.](?P<index>[0-9]+)[.]")
_SOURCE_BODY_LAYER = re.compile(r"(?:^|[.])layers[.](?P<index>[0-9]+)[.]")

_PLAN_KEYS = frozenset(
    {
        "schema",
        "state",
        "algorithm",
        "producer",
        "profile",
        "source",
        "probe",
        "model",
        "search",
        "tensor_metadata",
        "tensor_metadata_binding",
        "scales",
        "seams",
        "transforms",
        "verification",
        "plan_sha256",
    }
)
_PLAN_SET_KEYS = _PLAN_KEYS | {"workers"}
_MODEL_KEYS = frozenset(
    {"hidden_size", "layer_count", "planned_layers", "excluded_prefixes"}
)
_SEARCH_KEYS = frozenset(
    {
        "algorithm",
        "group_size",
        "alphas",
        "max_rounds",
        "variant",
        "polish_top",
        "polish_pool",
        "nvfp4_scale_rule",
        "nvfp4_snapped_scale_scoring",
        "nvfp4_joint_scale_levels",
        "objective_fold_dtype",
        "global_scale_scope",
        "materialization_rounding",
    }
)
_VERIFICATION_KEYS = frozenset(
    {
        "fp64_invariance_max_abs",
        "threshold",
        "domain",
        "required_bf16_fold_kl_max",
    }
)
_SCALE_METADATA_KEYS = frozenset({"file", "sha256", "vectors"})
_TENSOR_METADATA_KEYS = frozenset({"schema", "tensors", "sha256"})
_TENSOR_ROW_KEYS = frozenset({"owner", "shape", "dtype"})
_TENSOR_METADATA_BINDING_KEYS = frozenset(
    {"mode", "manifest_sha256", "tensor_metadata_sha256"}
)
_TENSOR_METADATA_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "state",
        "tensor_metadata",
        "source_local_content_sha256",
        "source_portable_content_sha256",
        "producer",
        "manifest_sha256",
    }
)
_SEAM_STATS_KEYS = frozenset(
    {
        "algorithm",
        "error_baseline",
        "error_final",
        "improvement_fraction",
        "groups",
        "groups_moved",
        "rounds",
        "candidate_count",
        "fell_back",
        "polish_pool",
        "polished",
        "variant",
    }
)
_NORM_SEAM_KEYS = frozenset(
    {
        "layer",
        "kind",
        "vector",
        "norm",
        "norm_parameter_offset",
        "consumers",
        "stats",
        "graph_sha256",
    }
)
_UP_DOWN_SEAM_KEYS = frozenset(
    {
        "layer",
        "kind",
        "vector",
        "gate",
        "up",
        "down",
        "stats",
        "graph_sha256",
    }
)
_PROBE_BINDING_KEYS = frozenset(
    {
        "schema",
        "state",
        "original_probe_path",
        "original_probe_sha256",
        "normalized_probe_path",
        "normalized_probe_sha256",
        "source_root",
        "source_local_content_sha256",
        "source_portable_content_sha256",
        "source_identity_file_sha256",
        "delta",
        "producer",
        "binding_sha256",
    }
)

_PART_KEYS = frozenset(
    {
        "schema",
        "state",
        "plan_sha256",
        "producer",
        "plan_source_local_content_sha256",
        "source_portable_content_sha256",
        "shards",
        "changed_tensors",
        "part_sha256",
    }
)
_SHARD_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "plan_sha256",
        "source_name",
        "source_bytes",
        "source_sha256",
        "output_bytes",
        "output_sha256",
        "tensor_count",
        "changed_tensors",
    }
)


def _sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    data = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    tmp = path.with_name(f".{path.name}.tmp")
    if os.path.lexists(tmp):
        raise RuntimeError(f"refusing stale PrismaSnap temporary file {tmp}")
    with tmp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_file_durable(source: Path, destination: Path) -> None:
    """Copy one regular file and durably publish its payload bytes."""
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"PrismaSnap metadata source is not a regular file: {source}")
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_file():
            raise RuntimeError(
                f"PrismaSnap copy destination is not a regular file: {destination}"
            )
        if (
            destination.stat().st_size != source.stat().st_size
            or _sha256_file(destination) != _sha256_file(source)
        ):
            raise RuntimeError(
                f"PrismaSnap existing copied file differs from source: {destination}"
            )
        return
    temporary = destination.with_name(f".{destination.name}.prismasnap-copy.tmp")
    if os.path.lexists(temporary):
        if temporary.is_symlink() or not temporary.is_file():
            raise RuntimeError(f"unsafe PrismaSnap copy temporary: {temporary}")
        # The destination is absent, so this can only be an unpublished copy
        # interrupted before atomic rename.  Recreate it deterministically.
        temporary.unlink()
    shutil.copy2(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _load_json(path: Path, *, where: str) -> dict[str, Any]:
    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-JSON constant {value}")

    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("not one regular file")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=exact_object,
            parse_constant=reject_constant,
        )
    except Exception as exc:
        raise RuntimeError(f"{where} {path} is unreadable/corrupt") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{where} {path} must contain a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], *, where: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise RuntimeError(
            f"{where} fields differ: missing={sorted(set(expected) - actual)} "
            f"extra={sorted(actual - set(expected))}"
        )


def _require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeError(f"{where} must be a full lowercase SHA-256")
    return value


def _require_nonnegative_int(value: object, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{where} must be a non-negative integer")
    return value


def _sealed_state(body: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(body)
    normalized["state_sha256"] = canonical_json_sha256(
        normalized, where="PrismaSnap merge state"
    )
    return normalized


def _validate_sealed_state(
    value: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    where: str,
) -> None:
    _require_exact_keys(value, frozenset(expected), where=where)
    claimed = _require_sha256(value.get("state_sha256"), where=f"{where}.state_sha256")
    unsigned = {
        str(key): item for key, item in value.items() if key != "state_sha256"
    }
    if claimed != canonical_json_sha256(unsigned, where=where):
        raise RuntimeError(f"{where} self digest mismatch")
    if dict(value) != dict(expected):
        raise RuntimeError(f"{where} belongs to different inputs")


def _plan_digest(payload: Mapping[str, object]) -> str:
    unsigned = {str(key): value for key, value in payload.items() if key != "plan_sha256"}
    return canonical_json_sha256(unsigned, where="PrismaSnap plan")


def _producer_identity() -> dict[str, object]:
    """Bind the implementation bytes even while preparing a release commit."""
    repository = Path(__file__).resolve().parents[1]
    declared_commit = os.environ.get("PRISMAQUANT_PRODUCER_GIT_COMMIT")
    if declared_commit is not None:
        declared_commit = declared_commit.lower()
        if re.fullmatch(r"[0-9a-f]{40}", declared_commit) is None:
            raise RuntimeError(
                "PRISMAQUANT_PRODUCER_GIT_COMMIT must be a full 40-hex object id"
            )
    live_commit: str | None = None
    try:
        live_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip().lower()
    except Exception:
        live_commit = None
    if declared_commit is not None and live_commit is not None and (
        declared_commit != live_commit
    ):
        raise RuntimeError(
            "PrismaSnap declared producer commit differs from the live checkout"
        )
    commit = declared_commit or live_commit
    if commit is None:
        raise RuntimeError(
            "PrismaSnap cannot resolve producer git commit; an immutable archive "
            "must set PRISMAQUANT_PRODUCER_GIT_COMMIT"
        )
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("PrismaSnap producer git commit is malformed")
    relevant = [
        repository / "prismaquant" / "prismasnap.py",
        repository / "prismaquant" / "prismasnap_checkpoint.py",
        repository / "prismaquant" / "prismasnap_validation.py",
        repository / "prismaquant" / "export_native_compressed.py",
        repository / "prismaquant" / "cost_stage_checkpoint.py",
        repository / "prismaquant" / "cost_streaming.py",
        repository / "tools" / "prismasnap.py",
    ]
    # Graph discovery is profile-driven.  Bind all profile implementation and
    # declarative spec bytes so an uncommitted profile edit cannot change the
    # meaning of a plan while leaving its producer receipt apparently stable.
    profile_root = repository / "prismaquant" / "model_profiles"
    relevant.extend(sorted(profile_root.glob("*.py")))
    relevant.extend(sorted((profile_root / "specs").glob("*.json")))
    files = {
        str(path.relative_to(repository)): _sha256_file(path)
        for path in relevant
    }
    source_sha256 = canonical_json_sha256(
        files, where="PrismaSnap producer source files"
    )
    container_rootfs = os.environ.get("PRISMAQUANT_CONTAINER_ROOTFS_SHA256")
    if container_rootfs is not None:
        container_rootfs = container_rootfs.lower()
        if _SHA256.fullmatch(container_rootfs) is None:
            raise RuntimeError(
                "PRISMAQUANT_CONTAINER_ROOTFS_SHA256 must be a full SHA-256"
            )
    return {
        "git_commit": commit,
        "source_sha256": source_sha256,
        "source_files": files,
        "container_rootfs_sha256": container_rootfs,
        "container_attested": container_rootfs is not None,
    }


def _importance_digest(value: object) -> str:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise RuntimeError(
            f"PrismaSnap activation importance must be rank 1, got {array.shape}"
        )
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


class _Checkpoint:
    def __init__(self, root: Path, *, require_all_shards: bool = True):
        requested_root = Path(root)
        if requested_root.is_symlink() or not requested_root.is_dir():
            raise RuntimeError(f"PrismaSnap source must be a real directory: {root}")
        self.root = requested_root.resolve(strict=True)
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_symlink() or not index_path.is_file():
            raise RuntimeError(
                "PrismaSnap production materializer requires an indexed "
                f"safetensors checkpoint: {index_path}"
            )
        index = _load_json(index_path, where="PrismaSnap safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("PrismaSnap source index has no weight_map")
        if not all(
            isinstance(key, str)
            and key
            and isinstance(shard, str)
            and shard
            and Path(shard).name == shard
            for key, shard in weight_map.items()
        ):
            raise RuntimeError("PrismaSnap source weight_map is malformed")
        self.index = index
        self.weight_map = dict(sorted(weight_map.items()))
        self.shards = sorted(set(self.weight_map.values()))
        self.available_shards = set()
        for name in self.shards:
            path = self.root / name
            if path.is_symlink():
                raise RuntimeError(f"PrismaSnap source shard may not be a symlink: {path}")
            if path.is_file():
                self.available_shards.add(name)
        missing = sorted(set(self.shards) - self.available_shards)
        if require_all_shards and missing:
            raise RuntimeError(
                f"PrismaSnap source index references missing shards {missing[:8]}"
            )
        self._metadata: dict[str, tuple[tuple[int, ...], str]] = {}
        self._verified_fingerprints: dict[str, tuple[int, int, int, int, int]] = {}

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"PrismaSnap source shard is not regular: {path}")
        row = path.stat()
        return (
            int(row.st_dev),
            int(row.st_ino),
            int(row.st_size),
            int(row.st_mtime_ns),
            int(row.st_ctime_ns),
        )

    def record_verified_shard(
        self, name: str, before: tuple[int, int, int, int, int]
    ) -> None:
        after = self._fingerprint(self.root / name)
        if after != before:
            raise RuntimeError(
                f"PrismaSnap source shard changed during verification: {name}"
            )
        self._verified_fingerprints[name] = after

    def _assert_shard_stable(self, name: str) -> None:
        expected = self._verified_fingerprints.get(name)
        if expected is not None and self._fingerprint(self.root / name) != expected:
            raise RuntimeError(f"PrismaSnap verified source shard changed: {name}")

    def verify_stable(self) -> None:
        for name in sorted(self._verified_fingerprints):
            self._assert_shard_stable(name)

    def require_shards(self, names: Sequence[str], *, where: str) -> None:
        requested = set(str(name) for name in names)
        unknown = requested - set(self.shards)
        if unknown:
            raise RuntimeError(f"{where} references unknown shards {sorted(unknown)}")
        missing = requested - self.available_shards
        if missing:
            raise RuntimeError(f"{where} lacks required local shards {sorted(missing)}")

    def metadata(self, key: str) -> tuple[tuple[int, ...], str]:
        cached = self._metadata.get(key)
        if cached is not None:
            return cached
        shard = self.weight_map.get(key)
        if shard is None:
            raise RuntimeError(f"PrismaSnap graph references absent tensor {key!r}")
        self.require_shards([shard], where=f"PrismaSnap tensor {key!r}")
        self._assert_shard_stable(shard)
        with safe_open(str(self.root / shard), framework="pt") as handle:
            if key not in handle.keys():
                raise RuntimeError(
                    f"PrismaSnap index maps {key!r} to {shard}, but the shard lacks it"
                )
            sliced = handle.get_slice(key)
            value = (tuple(map(int, sliced.get_shape())), str(sliced.get_dtype()))
        self._assert_shard_stable(shard)
        self._metadata[key] = value
        return value

    def load(self, key: str, device: torch.device) -> torch.Tensor:
        shard = self.weight_map.get(key)
        if shard is None:
            raise RuntimeError(f"PrismaSnap graph references absent tensor {key!r}")
        self.require_shards([shard], where=f"PrismaSnap tensor {key!r}")
        self._assert_shard_stable(shard)
        path = self.root / shard
        try:
            context = safe_open(str(path), framework="pt", device=str(device))
            direct = True
        except (RuntimeError, TypeError):
            context = safe_open(str(path), framework="pt")
            direct = False
        with context as handle:
            value = handle.get_tensor(key)
        self._assert_shard_stable(shard)
        if not direct:
            value = value.to(device, non_blocking=True)
        return value.contiguous()


def _scan_checkpoint_tensor_metadata(source: _Checkpoint) -> dict[str, object]:
    """Read one canonical, payload-free tensor census from all shard headers."""
    source.require_shards(
        source.shards, where="PrismaSnap full tensor-metadata header scan"
    )
    tensors: dict[str, dict[str, object]] = {}
    for shard in source.shards:
        expected = {
            key for key, owner in source.weight_map.items() if owner == shard
        }
        before = source._fingerprint(source.root / shard)
        source._assert_shard_stable(shard)
        with safe_open(str(source.root / shard), framework="pt") as handle:
            keys = set(handle.keys())
            if keys != expected:
                raise RuntimeError(
                    "PrismaSnap source header/index tensor census differs for "
                    f"{shard}: missing={sorted(expected - keys)[:12]} "
                    f"extra={sorted(keys - expected)[:12]}"
                )
            for key in sorted(keys):
                sliced = handle.get_slice(key)
                shape = tuple(map(int, sliced.get_shape()))
                dtype = str(sliced.get_dtype())
                source._metadata[key] = (shape, dtype)
                tensors[key] = {
                    "owner": shard,
                    "shape": list(shape),
                    "dtype": dtype,
                }
        source.record_verified_shard(shard, before)
    if set(tensors) != set(source.weight_map):
        raise RuntimeError("PrismaSnap full tensor-metadata census is incomplete")
    unsigned: dict[str, object] = {
        "schema": TENSOR_METADATA_SCHEMA,
        "tensors": dict(sorted(tensors.items())),
    }
    return {
        **unsigned,
        "sha256": canonical_json_sha256(
            unsigned, where="PrismaSnap full tensor metadata"
        ),
    }


def _validate_source_identity(
    source: _Checkpoint,
    identity_path: Path,
    *,
    verify_content: bool,
) -> tuple[dict[str, object], dict[str, object], str]:
    try:
        identity_bytes = identity_path.read_bytes()
        raw = json.loads(identity_bytes)
    except Exception as exc:
        raise RuntimeError(
            f"PrismaSnap source identity {identity_path} is unreadable/corrupt"
        ) from exc
    if not isinstance(raw, dict):
        raise RuntimeError("PrismaSnap source identity must be a JSON object")
    identity = validate_streamed_model_identity(
        raw, where="PrismaSnap source identity"
    )
    if identity.get("checkpoint_weight_map") != source.weight_map:
        raise RuntimeError(
            "PrismaSnap source checkpoint index differs from its identity"
        )
    _validate_config_semantics(source.root, identity)
    by_name = {Path(str(row["path"])).name: row for row in identity["shards"]}
    if set(by_name) != set(source.shards):
        raise RuntimeError(
            "PrismaSnap source identity shard census differs from checkpoint index"
        )
    for name in source.shards:
        path = source.root / name
        row = by_name[name]
        if name not in source.available_shards:
            continue
        fingerprint = source._fingerprint(path)
        if int(row["size"]) != fingerprint[2]:
            raise RuntimeError(f"PrismaSnap source shard size changed: {path}")
        if verify_content:
            actual = _sha256_file(path)
            if actual != row["sha256"]:
                raise RuntimeError(f"PrismaSnap source shard content changed: {path}")
        source.record_verified_shard(name, fingerprint)
    portable = portable_streamed_model_content_identity(
        identity, where="PrismaSnap portable source identity"
    )
    return identity, portable, hashlib.sha256(identity_bytes).hexdigest()


def _validate_config_semantics(
    model_root: Path,
    identity: Mapping[str, object],
) -> None:
    """Bind the live HF config semantics carried by the source identity."""
    config_path = model_root / "config.json"
    before = (config_path.stat().st_size, _sha256_file(config_path))
    try:
        from transformers import AutoConfig

        live = canonical_streamed_model_semantic_config(
            AutoConfig.from_pretrained(
                model_root,
                trust_remote_code=True,
                local_files_only=True,
            ).to_dict(),
            where="PrismaSnap live source config",
        )
        expected = canonical_streamed_model_semantic_config(
            identity.get("config"), where="PrismaSnap identity config"
        )
    except Exception as exc:
        raise RuntimeError(
            f"PrismaSnap cannot validate model config semantics at {model_root}"
        ) from exc
    after = (config_path.stat().st_size, _sha256_file(config_path))
    if before != after:
        raise RuntimeError("PrismaSnap model config changed during validation")
    if live != expected:
        raise RuntimeError(
            "PrismaSnap live model config semantics differ from its identity"
        )


def _load_probe(
    path: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, object], str]:
    # The probe is a trusted, locally produced PrismaQuant artifact.  It is
    # never accepted from an untrusted release/model directory.
    probe_bytes = path.read_bytes()
    probe = pickle.loads(probe_bytes)  # noqa: S301 - documented trusted artifact
    if not isinstance(probe, dict):
        raise RuntimeError("PrismaSnap probe is not a mapping")
    stats = probe.get("stats")
    meta = probe.get("meta")
    if not isinstance(stats, dict) or not stats:
        raise RuntimeError("PrismaSnap probe has no Linear stats")
    if not isinstance(meta, dict):
        raise RuntimeError("PrismaSnap probe has no metadata")
    required = {"act_sq_sum", "in_features", "out_features"}
    for qname, value in stats.items():
        if not isinstance(qname, str) or not isinstance(value, dict):
            raise RuntimeError("PrismaSnap probe stats are malformed")
        # MTP/lm_head bookkeeping rows can legitimately carry a different
        # estimator schema.  This planner's explicit scope is body decoder
        # layers; validate those rows strictly and leave excluded namespaces
        # untouched for the ordinary pipeline.
        if _BODY_LAYER.match(qname) is None:
            continue
        missing = required - set(value)
        if missing:
            raise RuntimeError(f"PrismaSnap probe row {qname!r} lacks {sorted(missing)}")
        importance = np.asarray(value["act_sq_sum"], dtype=np.float32)
        if (
            importance.ndim != 1
            or importance.size != int(value["in_features"])
            or not np.isfinite(importance).all()
            or np.any(importance < 0)
            or not np.any(importance > 0)
        ):
            raise RuntimeError(f"PrismaSnap probe row {qname!r} has invalid importance")
    return stats, meta, hashlib.sha256(probe_bytes).hexdigest()


def _validate_probe_source_contract(
    meta: Mapping[str, object], source: _Checkpoint
) -> None:
    model = meta.get("model")
    try:
        probe_model = Path(str(model)).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("PrismaSnap probe model path is not resolvable") from exc
    nsamples = meta.get("nsamples")
    seqlen = meta.get("seqlen")
    execution_device = meta.get("execution_device")
    if (
        not isinstance(model, str)
        or probe_model != source.root
        or meta.get("dtype") != "bf16"
        or not isinstance(execution_device, str)
        or re.fullmatch(r"cuda(?::[0-9]+)?", execution_device) is None
        or meta.get("device_map") != "streaming-layerwise"
        or type(nsamples) is not int
        or nsamples <= 0
        or type(seqlen) is not int
        or seqlen <= 0
        or meta.get("calibration_modality") != "text-only"
        or not isinstance(meta.get("dataset"), str)
        or not meta.get("dataset")
        or not isinstance(meta.get("calib_hash"), str)
        or not meta.get("calib_hash")
    ):
        raise RuntimeError(
            "PrismaSnap probe is not bound to the BF16 streaming-GPU text "
            "calibration contract for this source"
        )


def _probe_binding_path(probe_path: Path) -> Path:
    return probe_path.with_name(probe_path.name + PROBE_BINDING_SUFFIX)


def _legacy_normalized_probe_bytes(
    original_bytes: bytes, source: _Checkpoint
) -> tuple[bytes, str]:
    try:
        payload = pickle.loads(original_bytes)  # noqa: S301 - trusted local probe
    except Exception as exc:
        raise RuntimeError("PrismaSnap legacy probe pickle is corrupt") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("PrismaSnap legacy probe is not a mapping")
    stats = payload.get("stats")
    meta = payload.get("meta")
    if not isinstance(stats, dict) or not stats or not isinstance(meta, dict):
        raise RuntimeError("PrismaSnap legacy probe contract is malformed")
    modality = (
        "missing" if "calibration_modality" not in meta else meta["calibration_modality"]
    )
    model = meta.get("model")
    execution_device = meta.get("execution_device")
    try:
        model_root = Path(str(model)).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("PrismaSnap legacy probe model alias is invalid") from exc
    if (
        (modality == "missing" and "calibration_modality" in meta)
        or modality not in {"missing", "text-only"}
        or not isinstance(model, str)
        or model_root != source.root
        or meta.get("dtype") != "bf16"
        or meta.get("device_map") != "streaming-layerwise"
        or not isinstance(execution_device, str)
        or re.fullmatch(r"cuda(?::[0-9]+)?", execution_device) is None
        or type(meta.get("nsamples")) is not int
        or int(meta["nsamples"]) < 2
        or type(meta.get("seqlen")) is not int
        or int(meta["seqlen"]) < 512
        or not isinstance(meta.get("dataset"), str)
        or not meta.get("dataset")
        or not isinstance(meta.get("calib_hash"), str)
        or not meta.get("calib_hash")
    ):
        raise RuntimeError(
            "PrismaSnap legacy probe does not satisfy the frozen BF16 "
            "streaming-CUDA token-probe contract"
        )
    visual = [
        name
        for name in stats
        if isinstance(name, str)
        and (name.startswith("model.visual.") or name.startswith("visual."))
    ]
    if visual:
        raise RuntimeError(
            f"PrismaSnap legacy text probe contains visual stats {visual[:8]}"
        )
    normalized_meta = dict(meta)
    normalized_meta["calibration_modality"] = "text-only"
    normalized = dict(payload)
    normalized["meta"] = normalized_meta
    return pickle.dumps(normalized, protocol=pickle.HIGHEST_PROTOCOL), str(modality)


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if os.path.lexists(temporary):
        raise RuntimeError(f"refusing stale PrismaSnap temporary file {temporary}")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _validate_probe_binding_receipt(
    normalized_path: Path,
    *,
    source: _Checkpoint,
    identity: Mapping[str, object],
    portable: Mapping[str, object],
    identity_file_sha256: str,
    producer: Mapping[str, object],
) -> dict[str, object]:
    receipt_path = _probe_binding_path(normalized_path)
    receipt = _load_json(receipt_path, where="PrismaSnap legacy probe binding")
    _require_exact_keys(
        receipt, _PROBE_BINDING_KEYS, where="PrismaSnap legacy probe binding"
    )
    claimed = _require_sha256(
        receipt.get("binding_sha256"),
        where="PrismaSnap legacy probe binding.binding_sha256",
    )
    unsigned = {
        str(key): value
        for key, value in receipt.items()
        if key != "binding_sha256"
    }
    original_raw = receipt.get("original_probe_path")
    if not isinstance(original_raw, str):
        raise RuntimeError("PrismaSnap legacy probe original path is malformed")
    requested_original = Path(original_raw)
    if requested_original.is_symlink() or not requested_original.is_file():
        raise RuntimeError("PrismaSnap legacy probe original is not a regular file")
    original = requested_original.resolve(strict=True)
    normalized_bytes = normalized_path.read_bytes()
    original_bytes = original.read_bytes()
    expected_bytes, before_modality = _legacy_normalized_probe_bytes(
        original_bytes, source
    )
    expected_delta = {
        "field": "meta.calibration_modality",
        "before": before_modality,
        "after": "text-only",
        "only_mutation": True,
    }
    if (
        receipt.get("schema") != LEGACY_TEXT_PROBE_BINDING_SCHEMA
        or receipt.get("state") != "BOUND"
        or claimed
        != canonical_json_sha256(unsigned, where="PrismaSnap legacy probe binding")
        or original_raw != str(original)
        or receipt.get("normalized_probe_path") != str(normalized_path)
        or _require_sha256(
            receipt.get("original_probe_sha256"),
            where="PrismaSnap legacy probe binding.original_probe_sha256",
        )
        != hashlib.sha256(original_bytes).hexdigest()
        or _require_sha256(
            receipt.get("normalized_probe_sha256"),
            where="PrismaSnap legacy probe binding.normalized_probe_sha256",
        )
        != hashlib.sha256(normalized_bytes).hexdigest()
        or normalized_bytes != expected_bytes
        or receipt.get("source_root") != str(source.root)
        or receipt.get("source_local_content_sha256") != identity.get("content_sha256")
        or receipt.get("source_portable_content_sha256")
        != portable.get("portable_content_sha256")
        or receipt.get("source_identity_file_sha256") != identity_file_sha256
        or receipt.get("delta") != expected_delta
        or receipt.get("producer") != producer
    ):
        raise RuntimeError("PrismaSnap legacy probe binding contract failed")
    return receipt


def bind_legacy_text_probe(
    source_dir: str | Path,
    source_identity_path: str | Path,
    probe_path: str | Path,
    output_path: str | Path,
    *,
    verify_source_content: bool = True,
    resume: bool = False,
    production: bool = False,
) -> dict[str, object]:
    """Normalize only the missing legacy text-modality field with a receipt."""
    if production and not verify_source_content:
        raise RuntimeError(
            "PrismaSnap production probe binding cannot skip source verification"
        )
    source = _Checkpoint(Path(source_dir))
    identity_path = Path(source_identity_path).resolve(strict=True)
    identity, portable, identity_file_sha256 = _validate_source_identity(
        source, identity_path, verify_content=verify_source_content
    )
    requested_probe = Path(probe_path)
    if requested_probe.is_symlink() or not requested_probe.is_file():
        raise RuntimeError("PrismaSnap legacy probe must be a regular local file")
    original = requested_probe.resolve(strict=True)
    _load_probe(original)
    requested_output = Path(output_path).absolute()
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    output = requested_output.parent.resolve(strict=True) / requested_output.name
    if output == original:
        raise RuntimeError("PrismaSnap normalized probe output must be distinct")
    normalized_bytes, before_modality = _legacy_normalized_probe_bytes(
        original.read_bytes(), source
    )
    producer = _producer_identity()
    if production and not producer.get("container_attested"):
        raise RuntimeError(
            "PrismaSnap production probe binding requires an attested container"
        )
    delta = {
        "field": "meta.calibration_modality",
        "before": before_modality,
        "after": "text-only",
        "only_mutation": True,
    }
    unsigned: dict[str, object] = {
        "schema": LEGACY_TEXT_PROBE_BINDING_SCHEMA,
        "state": "BOUND",
        "original_probe_path": str(original),
        "original_probe_sha256": _sha256_file(original),
        "normalized_probe_path": str(output),
        "normalized_probe_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        "source_root": str(source.root),
        "source_local_content_sha256": identity["content_sha256"],
        "source_portable_content_sha256": portable["portable_content_sha256"],
        "source_identity_file_sha256": identity_file_sha256,
        "delta": delta,
        "producer": producer,
    }
    expected = {
        **unsigned,
        "binding_sha256": canonical_json_sha256(
            unsigned, where="PrismaSnap legacy probe binding"
        ),
    }
    source.verify_stable()
    receipt_path = _probe_binding_path(output)
    output_exists = os.path.lexists(output)
    receipt_exists = os.path.lexists(receipt_path)
    if output_exists:
        if output.is_symlink() or not output.is_file() or not resume:
            raise RuntimeError(f"PrismaSnap normalized probe output exists: {output}")
        if output.read_bytes() != normalized_bytes:
            raise RuntimeError("PrismaSnap normalized probe output is equivocal")
        if receipt_exists:
            observed = _validate_probe_binding_receipt(
                output,
                source=source,
                identity=identity,
                portable=portable,
                identity_file_sha256=identity_file_sha256,
                producer=producer,
            )
            if observed != expected:
                raise RuntimeError("PrismaSnap legacy probe binding is equivocal")
            source.verify_stable()
            return observed
        _atomic_json(receipt_path, expected)
    else:
        if receipt_exists:
            raise RuntimeError("PrismaSnap legacy probe receipt lacks its output")
        _atomic_bytes(output, normalized_bytes)
        _atomic_json(receipt_path, expected)
    observed = _validate_probe_binding_receipt(
        output,
        source=source,
        identity=identity,
        portable=portable,
        identity_file_sha256=identity_file_sha256,
        producer=producer,
    )
    if observed != expected:
        raise RuntimeError("PrismaSnap legacy probe changed after binding")
    source.verify_stable()
    return observed


def _validate_tensor_metadata_manifest(
    manifest_path: Path,
    *,
    identity: Mapping[str, object],
    portable: Mapping[str, object],
    producer: Mapping[str, object],
) -> dict[str, object]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("PrismaSnap tensor-metadata manifest is not a regular file")
    payload = _load_json(
        manifest_path, where="PrismaSnap tensor-metadata manifest"
    )
    _require_exact_keys(
        payload,
        _TENSOR_METADATA_MANIFEST_KEYS,
        where="PrismaSnap tensor-metadata manifest",
    )
    claimed = _require_sha256(
        payload.get("manifest_sha256"),
        where="PrismaSnap tensor-metadata manifest.manifest_sha256",
    )
    unsigned = {
        str(key): value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if (
        payload.get("schema") != TENSOR_METADATA_MANIFEST_SCHEMA
        or payload.get("state") != "SCANNED"
        or claimed
        != canonical_json_sha256(unsigned, where="PrismaSnap tensor-metadata manifest")
        or _require_sha256(
            payload.get("source_local_content_sha256"),
            where="PrismaSnap tensor-metadata manifest source local identity",
        )
        != payload.get("source_local_content_sha256")
        or payload.get("source_portable_content_sha256")
        != portable.get("portable_content_sha256")
        or payload.get("producer") != producer
    ):
        raise RuntimeError("PrismaSnap tensor-metadata manifest contract failed")
    # The portable identity is the cross-host equality contract.  Validate the
    # full tensor census against this worker's independently admitted index.
    _validate_tensor_metadata_contract(
        {
            "source": {"identity": identity},
            "tensor_metadata": payload.get("tensor_metadata"),
        }
    )
    return payload


def scan_tensor_metadata_manifest(
    source_dir: str | Path,
    source_identity_path: str | Path,
    output_path: str | Path,
    *,
    verify_source_content: bool = True,
    resume: bool = False,
    production: bool = False,
) -> dict[str, object]:
    """Scan every safetensors header once and publish a portable census."""
    if production and not verify_source_content:
        raise RuntimeError(
            "PrismaSnap production header scan cannot skip source verification"
        )
    source = _Checkpoint(Path(source_dir))
    identity, portable, _identity_file_sha256 = _validate_source_identity(
        source,
        Path(source_identity_path).resolve(strict=True),
        verify_content=verify_source_content,
    )
    producer = _producer_identity()
    if production and not producer.get("container_attested"):
        raise RuntimeError(
            "PrismaSnap production header scan requires an attested container"
        )
    tensor_metadata = _scan_checkpoint_tensor_metadata(source)
    unsigned: dict[str, object] = {
        "schema": TENSOR_METADATA_MANIFEST_SCHEMA,
        "state": "SCANNED",
        "tensor_metadata": tensor_metadata,
        "source_local_content_sha256": identity["content_sha256"],
        "source_portable_content_sha256": portable["portable_content_sha256"],
        "producer": producer,
    }
    expected = {
        **unsigned,
        "manifest_sha256": canonical_json_sha256(
            unsigned, where="PrismaSnap tensor-metadata manifest"
        ),
    }
    source.verify_stable()
    requested_output = Path(output_path).absolute()
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    output = requested_output.parent.resolve(strict=True) / requested_output.name
    if os.path.lexists(output):
        if not resume or output.is_symlink() or not output.is_file():
            raise RuntimeError(
                f"PrismaSnap tensor-metadata manifest output exists: {output}"
            )
        observed = _validate_tensor_metadata_manifest(
            output, identity=identity, portable=portable, producer=producer
        )
        if observed != expected:
            raise RuntimeError("PrismaSnap tensor-metadata manifest is equivocal")
        source.verify_stable()
        return observed
    _atomic_json(output, expected)
    observed = _validate_tensor_metadata_manifest(
        output, identity=identity, portable=portable, producer=producer
    )
    if observed != expected:
        raise RuntimeError("PrismaSnap tensor-metadata manifest changed after scan")
    source.verify_stable()
    return observed


def _require_production_execution(device: str) -> None:
    try:
        execution_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError("PrismaSnap production device is malformed") from exc
    if execution_device.type != "cuda":
        raise RuntimeError("PrismaSnap production execution requires CUDA")
    if not bool(_producer_identity().get("container_attested")):
        raise RuntimeError(
            "PrismaSnap production execution requires an attested container rootfs"
        )


def _source_weight_key(profile, qname: str) -> str:
    return profile.source_tensor_name(f"{qname}.weight")


def _leaf(qname: str) -> str:
    return qname.rsplit(".", 1)[-1]


def _ordered_dense_consumers(names: Sequence[str], profile) -> list[str]:
    """Use declared fused-member order, then a deterministic lexical tail.

    The measured dense pilot accumulated Q/K/V in Q,K,V order.  Sorting raw
    names changes floating-point accumulation (K,Q,V) and therefore changes
    the selected plan.  Declarative profile groups also give hybrid DeltaNet
    its stable qkv,z then b,a extension without guessing from filesystem order.
    """
    remaining = set(str(name) for name in names)
    if len(remaining) != len(names):
        raise RuntimeError("PrismaSnap consumer graph repeats a Linear")
    ordered: list[str] = []
    for group_leaf, member_leaves in profile.fused_sibling_leaf_mapping().items():
        group_members = {
            name
            for name in remaining
            if (
                (group := profile.fused_sibling_group(name)) is not None
                and _leaf(str(group)) == str(group_leaf)
            )
        }
        if not group_members:
            continue
        by_leaf = {_leaf(name): name for name in group_members}
        if set(by_leaf) != set(member_leaves):
            raise RuntimeError(
                "PrismaSnap fused consumer graph is a partial/ambiguous group: "
                f"{sorted(group_members)}"
            )
        for leaf in member_leaves:
            ordered.append(by_leaf[str(leaf)])
        remaining.difference_update(group_members)
    ordered.extend(sorted(remaining))
    return ordered


_DENSE_ROLE_ORDER = {
    leaf: ordinal
    for ordinal, leaf in enumerate(
        (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_b.weight",
            "in_proj_a.weight",
            "gate_proj.weight",
            "up_proj.weight",
        )
    )
}


def _dense_source_role_order(name: str) -> tuple[int, str]:
    leaf = ".".join(name.rsplit(".", 2)[-2:])
    return (_DENSE_ROLE_ORDER.get(leaf, len(_DENSE_ROLE_ORDER)), name)


def _dense_plan_graph_sha256(
    *,
    layer: int,
    input_norm: str,
    input_offset: float,
    input_consumers: Sequence[str],
    post_norm: str,
    post_offset: float,
    post_consumers: Sequence[str],
    gate: str,
    up: str,
    down: str,
) -> str:
    return canonical_json_sha256(
        {
            "layer": layer,
            "input_norm": input_norm,
            "input_norm_parameter_offset": input_offset,
            "input_consumers": list(input_consumers),
            "post_attention_norm": post_norm,
            "post_attention_norm_parameter_offset": post_offset,
            "post_attention_consumers": list(post_consumers),
            "gate": gate,
            "up": up,
            "down": down,
        },
        where=f"PrismaSnap layer {layer} executable graph",
    )


def _discover_dense_layer_graph(
    layer_index: int,
    *,
    hidden_size: int,
    stats: Mapping[str, dict[str, object]],
    profile,
    checkpoint: _Checkpoint,
) -> dict[str, object]:
    """Discover direct consumers from the probe's activation equivalence graph.

    Linears that consumed the exact same runtime tensor have byte-identical
    ``act_sq_sum`` reductions.  We use those equivalence classes plus the
    profile's vLLM fused-module graph and checkpoint shapes.  This avoids a
    projection-name allowlist and fails closed on every unaccounted
    hidden-width consumer.
    """
    prefix = f"model.layers.{layer_index}."
    layer_stats = {
        qname: row for qname, row in stats.items() if qname.startswith(prefix)
    }
    if not layer_stats:
        raise RuntimeError(f"PrismaSnap probe has no rows for body layer {layer_index}")
    hidden_consumers = {
        qname: row
        for qname, row in layer_stats.items()
        if int(row["in_features"]) == hidden_size
    }
    clusters: dict[str, list[str]] = defaultdict(list)
    for qname, row in hidden_consumers.items():
        clusters[_importance_digest(row["act_sq_sum"])].append(qname)

    fused: dict[str, list[str]] = defaultdict(list)
    for qname in hidden_consumers:
        group = profile.fused_sibling_group(qname)
        if group is not None:
            fused[str(group)].append(qname)

    mlp_group: str | None = None
    mlp_members: list[str] | None = None
    down_qname: str | None = None
    for group, members in sorted(fused.items()):
        if len(members) != 2:
            continue
        output_widths = {int(layer_stats[name]["out_features"]) for name in members}
        if len(output_widths) != 1:
            continue
        intermediate = next(iter(output_widths))
        downstream = [
            qname
            for qname, row in layer_stats.items()
            if qname not in members
            and int(row["in_features"]) == intermediate
            and int(row["out_features"]) == hidden_size
        ]
        if len(downstream) == 1:
            if mlp_group is not None:
                raise RuntimeError(
                    f"layer {layer_index}: multiple candidate dense MLP graphs"
                )
            mlp_group, mlp_members, down_qname = group, sorted(members), downstream[0]
    if mlp_group is None or mlp_members is None or down_qname is None:
        raise RuntimeError(f"layer {layer_index}: could not discover dense MLP graph")

    post_digest = _importance_digest(layer_stats[mlp_members[0]]["act_sq_sum"])
    if sorted(clusters.get(post_digest, [])) != sorted(mlp_members):
        raise RuntimeError(
            f"layer {layer_index}: post-norm activation class has unknown consumers "
            f"{sorted(clusters.get(post_digest, []))}"
        )
    input_clusters = {
        digest: names for digest, names in clusters.items() if digest != post_digest
    }
    if len(input_clusters) != 1:
        raise RuntimeError(
            f"layer {layer_index}: expected one token-mixer input class, got "
            f"{[sorted(v) for v in input_clusters.values()]}"
        )
    input_digest, unordered_input_members = next(iter(input_clusters.items()))
    input_members = _ordered_dense_consumers(unordered_input_members, profile)
    if len(input_members) < 2:
        raise RuntimeError(
            f"layer {layer_index}: token-mixer graph has too few direct consumers"
        )

    leaf_mapping = profile.fused_sibling_leaf_mapping()
    group_leaf = _leaf(mlp_group)
    ordered_leaves = leaf_mapping.get(group_leaf)
    if ordered_leaves is None or len(ordered_leaves) != 2:
        raise RuntimeError(
            f"layer {layer_index}: profile does not expose ordered MLP fused members"
        )
    by_leaf = {_leaf(name): name for name in mlp_members}
    if set(by_leaf) != set(ordered_leaves):
        raise RuntimeError(
            f"layer {layer_index}: observed MLP members {sorted(by_leaf)} differ "
            f"from profile graph {list(ordered_leaves)}"
        )
    gate_qname = by_leaf[ordered_leaves[0]]
    up_qname = by_leaf[ordered_leaves[1]]
    mlp_members = [by_leaf[leaf] for leaf in ordered_leaves]

    declared_norm_offset = profile.rms_norm_parameter_offset()
    if declared_norm_offset is None:
        raise RuntimeError(
            f"layer {layer_index}: profile {profile.name!r} does not declare "
            "its RMSNorm parameter encoding"
        )
    norm_parameter_offset = float(declared_norm_offset)
    if not math.isfinite(norm_parameter_offset):
        raise RuntimeError(
            f"layer {layer_index}: profile returned a non-finite RMSNorm offset"
        )
    source_prefix = profile.source_tensor_name(prefix[:-1]) + "."
    input_norm = source_prefix + "input_layernorm.weight"
    post_norm = source_prefix + "post_attention_layernorm.weight"
    for norm in (input_norm, post_norm):
        shape, dtype = checkpoint.metadata(norm)
        if shape != (hidden_size,) or dtype != "BF16":
            raise RuntimeError(
                f"layer {layer_index}: invalid PrismaSnap norm {norm} shape/dtype "
                f"{shape}/{dtype}"
            )

    all_qnames = sorted(set(input_members) | set(mlp_members) | {down_qname})
    source_weights: dict[str, str] = {}
    for qname in all_qnames:
        key = _source_weight_key(profile, qname)
        shape, dtype = checkpoint.metadata(key)
        expected = (
            int(layer_stats[qname]["out_features"]),
            int(layer_stats[qname]["in_features"]),
        )
        if shape != expected:
            raise RuntimeError(
                f"layer {layer_index}: source/probe shape mismatch for {qname}: "
                f"{shape} != {expected}"
            )
        if dtype != "BF16":
            raise RuntimeError(
                f"layer {layer_index}: PrismaSnap production source must be BF16; "
                f"{key} is {dtype}. Native FP8 source refolding is refused."
            )
        source_weights[qname] = key

    graph_value = {
        "layer": layer_index,
        "input_activation_sha256": input_digest,
        "post_activation_sha256": post_digest,
        "input_norm": input_norm,
        "post_norm": post_norm,
        "norm_parameter_offset": norm_parameter_offset,
        "input_consumers": input_members,
        "post_consumers": mlp_members,
        "mlp_fused_group": mlp_group,
        "gate": gate_qname,
        "up": up_qname,
        "down": down_qname,
        "source_weights": source_weights,
    }
    graph_value["graph_sha256"] = canonical_json_sha256(
        graph_value, where=f"PrismaSnap layer {layer_index} graph"
    )
    return graph_value


def _importance(row: Mapping[str, object], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(row["act_sq_sum"], dtype=np.float32), device=device
    )


def _fp64_invariance_error(
    graph: Mapping[str, object],
    weights: Mapping[str, torch.Tensor],
    input_norm: torch.Tensor,
    post_norm: torch.Tensor,
    input_scale: torch.Tensor,
    post_scale: torch.Tensor,
    updown_scale: torch.Tensor,
) -> float:
    """Algebraic unit gate on real checkpoint values, sampled deterministically."""
    worst = 0.0
    norm_parameter_offset = float(graph["norm_parameter_offset"])
    for norm, scale, names in (
        (input_norm, input_scale, graph["input_consumers"]),
        (post_norm, post_scale, graph["post_consumers"]),
    ):
        parameter = norm.to(torch.float64)
        gamma = parameter + norm_parameter_offset
        folded_parameter = (
            gamma * scale.to(torch.float64) - norm_parameter_offset
        )
        folded_gamma = folded_parameter + norm_parameter_offset
        for name in names:
            weight = weights[str(name)].to(torch.float64)
            rows = torch.linspace(
                0, weight.shape[0] - 1, steps=min(7, weight.shape[0]),
                device=weight.device,
            ).round().long()
            cols = torch.linspace(
                0, weight.shape[1] - 1, steps=min(17, weight.shape[1]),
                device=weight.device,
            ).round().long()
            before = weight[rows][:, cols] * gamma[cols].view(1, -1)
            after = (
                weight[rows][:, cols] / scale[cols].view(1, -1)
            ) * folded_gamma[cols].view(1, -1)
            worst = max(worst, float((before - after).abs().max().item()))
    up = weights[str(graph["up"])].to(torch.float64)
    down = weights[str(graph["down"])].to(torch.float64)
    mids = torch.linspace(
        0, up.shape[0] - 1, steps=min(17, up.shape[0]), device=up.device
    ).round().long()
    ins = torch.linspace(
        0, up.shape[1] - 1, steps=min(7, up.shape[1]), device=up.device
    ).round().long()
    outs = torch.linspace(
        0, down.shape[0] - 1, steps=min(7, down.shape[0]), device=up.device
    ).round().long()
    before = down[outs][:, mids].unsqueeze(2) * up[mids][:, ins].unsqueeze(0)
    after = (
        down[outs][:, mids] / updown_scale[mids].view(1, -1)
    ).unsqueeze(2) * (
        up[mids][:, ins] * updown_scale[mids].view(-1, 1)
    ).unsqueeze(0)
    worst = max(worst, float((before - after).abs().max().item()))
    if not math.isfinite(worst) or worst > 1e-10:
        raise RuntimeError(f"PrismaSnap fp64 invariance gate failed: {worst:.3e}")
    return worst


def plan_dense_checkpoint(
    source_dir: str | Path,
    probe_path: str | Path,
    source_identity_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    layers: Sequence[int] | None = None,
    search_config: PrismaSnapSearchConfig | None = None,
    tensor_metadata_manifest_path: str | Path | None = None,
    verify_source_content: bool = True,
    resume: bool = False,
    production: bool = False,
) -> dict[str, object]:
    """Build a content-bound dense-body PrismaSnap plan, one layer at a time."""
    if production:
        _require_production_execution(device)
        if not verify_source_content:
            raise RuntimeError(
                "PrismaSnap production planning cannot skip source-content verification"
            )
    source = _Checkpoint(Path(source_dir), require_all_shards=False)
    identity, portable, identity_file_sha256 = _validate_source_identity(
        source,
        Path(source_identity_path).resolve(strict=True),
        verify_content=verify_source_content,
    )
    requested_probe = Path(probe_path)
    if requested_probe.is_symlink() or not requested_probe.is_file():
        raise RuntimeError("PrismaSnap probe must be a regular local file")
    probe_file = requested_probe.resolve(strict=True)
    stats, probe_meta, probe_sha256 = _load_probe(probe_file)
    _validate_probe_source_contract(probe_meta, source)
    config_payload = _load_json(source.root / "config.json", where="model config")
    hidden_size = int(config_payload.get("hidden_size", 0))
    layer_count = int(config_payload.get("num_hidden_layers", 0))
    if hidden_size <= 0 or layer_count <= 0:
        raise RuntimeError("PrismaSnap source config lacks hidden/layer count")
    requested_layers = list(range(layer_count)) if layers is None else sorted(set(layers))
    if not requested_layers or requested_layers[0] < 0 or requested_layers[-1] >= layer_count:
        raise ValueError("PrismaSnap requested layer set is empty or out of range")
    profile = detect_profile(str(source.root))
    if "dense" not in profile.name:
        raise RuntimeError(
            f"plan_dense_checkpoint refuses non-dense profile {profile.name!r}; "
            "use the MoE planner"
        )
    cfg = search_config or PrismaSnapSearchConfig()
    if production and cfg.as_dict() != PrismaSnapSearchConfig().as_dict():
        raise RuntimeError(
            "PrismaSnap production planning requires the measured-fast "
            "stage,polish treatment"
        )
    execution_device = torch.device(device)
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PrismaSnap production planning requested CUDA, but none is available")

    producer = _producer_identity()
    binding_path = _probe_binding_path(probe_file)
    probe_binding: dict[str, object] | None = None
    if os.path.lexists(binding_path):
        if binding_path.is_symlink() or not binding_path.is_file():
            raise RuntimeError("PrismaSnap probe has an unsafe binding receipt")
        probe_binding = _validate_probe_binding_receipt(
            probe_file,
            source=source,
            identity=identity,
            portable=portable,
            identity_file_sha256=identity_file_sha256,
            producer=producer,
        )
    elif production:
        raise RuntimeError(
            "PrismaSnap production planning requires a bound probe receipt"
        )
    if tensor_metadata_manifest_path is None:
        tensor_metadata = _scan_checkpoint_tensor_metadata(source)
        tensor_metadata_binding: dict[str, object] = {
            "mode": "inline_full_header_scan",
            "manifest_sha256": None,
            "tensor_metadata_sha256": tensor_metadata["sha256"],
        }
    else:
        manifest = _validate_tensor_metadata_manifest(
            Path(tensor_metadata_manifest_path).resolve(strict=True),
            identity=identity,
            portable=portable,
            producer=producer,
        )
        tensor_metadata = dict(manifest["tensor_metadata"])
        tensor_metadata_binding = {
            "mode": "external_manifest",
            "manifest_sha256": manifest["manifest_sha256"],
            "tensor_metadata_sha256": tensor_metadata["sha256"],
        }

    def validate_requested_plan(candidate: Mapping[str, object]) -> None:
        candidate_source = candidate.get("source")
        candidate_probe = candidate.get("probe")
        candidate_model = candidate.get("model")
        if (
            candidate.get("producer") != producer
            or candidate.get("profile") != profile.name
            or candidate.get("search") != cfg.as_dict()
            or candidate.get("tensor_metadata") != tensor_metadata
            or candidate.get("tensor_metadata_binding") != tensor_metadata_binding
            or not isinstance(candidate_source, Mapping)
            or candidate_source.get("portable_identity") != portable
            or not isinstance(candidate_probe, Mapping)
            or candidate_probe.get("sha256") != probe_sha256
            or not isinstance(candidate_model, Mapping)
            or candidate_model.get("planned_layers") != requested_layers
            or int(candidate_model.get("layer_count", -1)) != layer_count
            or int(candidate_model.get("hidden_size", -1)) != hidden_size
        ):
            raise RuntimeError(
                "PrismaSnap resumed plan belongs to different requested inputs"
            )

    output = Path(output_dir)
    if os.path.lexists(output):
        if resume and not output.is_symlink() and output.is_dir():
            existing, _scales = load_plan(output)
            validate_requested_plan(existing)
            return existing
        raise RuntimeError(f"PrismaSnap plan output already exists: {output}")
    staging = output.with_name(output.name + ".prismasnap-plan-incomplete")
    if os.path.lexists(staging):
        if not resume or staging.is_symlink() or not staging.is_dir():
            raise RuntimeError(f"stale PrismaSnap plan state exists: {staging}")
        if (staging / PLAN_JSON).is_file() and (staging / PLAN_SCALES).is_file():
            existing, _scales = load_plan(staging)
            validate_requested_plan(existing)
            os.replace(staging, output)
            _fsync_dir(output.parent)
            return existing
        allowed = {
            PLAN_SCALES,
            f".{PLAN_JSON}.tmp",
        }
        entries = list(staging.iterdir())
        if any(item.name not in allowed or item.is_symlink() for item in entries):
            raise RuntimeError(
                f"PrismaSnap plan staging has unexpected recovery state: {staging}"
            )
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        scale_tensors: dict[str, torch.Tensor] = {}
        seam_records: list[dict[str, object]] = []
        transforms: list[dict[str, object]] = []
        worst_invariance = 0.0
        for ordinal, layer_index in enumerate(requested_layers, start=1):
            print(
                f"[prismasnap-plan] layer {layer_index} "
                f"({ordinal}/{len(requested_layers)}) graph/load/search",
                flush=True,
            )
            graph = _discover_dense_layer_graph(
                layer_index,
                hidden_size=hidden_size,
                stats=stats,
                profile=profile,
                checkpoint=source,
            )
            qnames = sorted(graph["source_weights"])
            weights = {
                qname: source.load(str(graph["source_weights"][qname]), execution_device)
                for qname in qnames
            }
            input_norm = source.load(str(graph["input_norm"]), execution_device)
            post_norm = source.load(str(graph["post_norm"]), execution_device)

            def consumer(qname: str, mode: str) -> PrismaSnapConsumer:
                return PrismaSnapConsumer(
                    name=qname,
                    weight=weights[qname],
                    importance=_importance(stats[qname], execution_device),
                    mode=mode,  # type: ignore[arg-type]
                )

            def mean_importance(names: Sequence[str]) -> torch.Tensor:
                values = [_importance(stats[name], execution_device) for name in names]
                return torch.stack(values, dim=0).mean(dim=0)

            input_names = [str(name) for name in graph["input_consumers"]]
            post_names = [str(name) for name in graph["post_consumers"]]
            norm_parameter_offset = float(graph["norm_parameter_offset"])
            input_scale, input_stats = search_diagonal_scale(
                [consumer(name, "column_inverse") for name in input_names],
                mean_importance(input_names),
                config=cfg,
            )
            post_scale, post_stats = search_diagonal_scale(
                [consumer(name, "column_inverse") for name in post_names],
                mean_importance(post_names),
                config=cfg,
            )
            gate = str(graph["gate"])
            up = str(graph["up"])
            down = str(graph["down"])
            updown_scale, updown_stats = search_diagonal_scale(
                [
                    consumer(down, "column_inverse"),
                    consumer(up, "row"),
                ],
                _importance(stats[down], execution_device),
                config=cfg,
            )
            invariance = _fp64_invariance_error(
                graph,
                weights,
                input_norm,
                post_norm,
                input_scale,
                post_scale,
                updown_scale,
            )
            worst_invariance = max(worst_invariance, invariance)
            vector_names = {
                "input": f"layer_{layer_index:05d}_input",
                "post": f"layer_{layer_index:05d}_post",
                "updown": f"layer_{layer_index:05d}_updown",
            }
            scale_tensors[vector_names["input"]] = input_scale.cpu()
            scale_tensors[vector_names["post"]] = post_scale.cpu()
            scale_tensors[vector_names["updown"]] = updown_scale.cpu()
            input_source_consumers = [
                str(graph["source_weights"][name]) for name in input_names
            ]
            post_source_consumers = [
                str(graph["source_weights"][name]) for name in post_names
            ]
            source_gate = str(graph["source_weights"][gate])
            source_up = str(graph["source_weights"][up])
            source_down = str(graph["source_weights"][down])
            executable_graph_sha256 = _dense_plan_graph_sha256(
                layer=layer_index,
                input_norm=str(graph["input_norm"]),
                input_offset=norm_parameter_offset,
                input_consumers=input_source_consumers,
                post_norm=str(graph["post_norm"]),
                post_offset=norm_parameter_offset,
                post_consumers=post_source_consumers,
                gate=source_gate,
                up=source_up,
                down=source_down,
            )
            seam_records.extend(
                [
                    {
                        "layer": layer_index,
                        "kind": "input_norm",
                        "vector": vector_names["input"],
                        "norm": graph["input_norm"],
                        "norm_parameter_offset": norm_parameter_offset,
                        "consumers": input_source_consumers,
                        "stats": input_stats,
                        "graph_sha256": executable_graph_sha256,
                    },
                    {
                        "layer": layer_index,
                        "kind": "post_attention_norm",
                        "vector": vector_names["post"],
                        "norm": graph["post_norm"],
                        "norm_parameter_offset": norm_parameter_offset,
                        "consumers": post_source_consumers,
                        "stats": post_stats,
                        "graph_sha256": executable_graph_sha256,
                    },
                    {
                        "layer": layer_index,
                        "kind": "up_down",
                        "vector": vector_names["updown"],
                        "gate": source_gate,
                        "up": source_up,
                        "down": source_down,
                        "stats": updown_stats,
                        "graph_sha256": executable_graph_sha256,
                    },
                ]
            )
            transforms.append(
                {
                    "tensor": graph["input_norm"],
                    "vector": vector_names["input"],
                    "operation": "affine_multiply",
                    "axis": 0,
                    "order": 0,
                    "parameter_offset": norm_parameter_offset,
                }
            )
            transforms.extend(
                {
                    "tensor": graph["source_weights"][name],
                    "vector": vector_names["input"],
                    "operation": "divide",
                    "axis": 1,
                    "order": 0,
                }
                for name in input_names
            )
            transforms.append(
                {
                    "tensor": graph["post_norm"],
                    "vector": vector_names["post"],
                    "operation": "affine_multiply",
                    "axis": 0,
                    "order": 0,
                    "parameter_offset": norm_parameter_offset,
                }
            )
            transforms.extend(
                {
                    "tensor": graph["source_weights"][name],
                    "vector": vector_names["post"],
                    "operation": "divide",
                    "axis": 1,
                    "order": 0,
                }
                for name in post_names
            )
            transforms.extend(
                [
                    {
                        "tensor": graph["source_weights"][up],
                        "vector": vector_names["updown"],
                        "operation": "multiply",
                        "axis": 0,
                        # The measured v1 materializer rounded the post-norm
                        # inverse fold to BF16 before applying this row fold.
                        "order": 1,
                    },
                    {
                        "tensor": graph["source_weights"][down],
                        "vector": vector_names["updown"],
                        "operation": "divide",
                        "axis": 1,
                        "order": 0,
                    },
                ]
            )
            del weights, input_norm, post_norm

        save_file(
            {name: tensor.contiguous() for name, tensor in sorted(scale_tensors.items())},
            str(staging / PLAN_SCALES),
            metadata={"format": "pt", "algorithm": PRISMASNAP_ALGORITHM},
        )
        with (staging / PLAN_SCALES).open("rb") as handle:
            os.fsync(handle.fileno())
        scales_sha256 = _sha256_file(staging / PLAN_SCALES)
        plan: dict[str, object] = {
            "schema": PLAN_SCHEMA,
            "state": "PLANNED",
            "algorithm": PRISMASNAP_ALGORITHM,
            "producer": producer,
            "profile": profile.name,
            "source": {
                "identity": identity,
                "portable_identity": portable,
                "identity_file_sha256": identity_file_sha256,
            },
            "probe": {
                "path": str(probe_file),
                "sha256": probe_sha256,
                "calib_hash": probe_meta.get("calib_hash"),
                "dataset": probe_meta.get("dataset"),
                "nsamples": probe_meta.get("nsamples"),
                "seqlen": probe_meta.get("seqlen"),
                "calibration_modality": probe_meta.get("calibration_modality"),
                "model": probe_meta.get("model"),
                "dtype": probe_meta.get("dtype"),
                "device_map": probe_meta.get("device_map"),
                "execution_device": probe_meta.get("execution_device"),
                "legacy_text_binding": (
                    None
                    if probe_binding is None
                    else {
                        "binding_sha256": probe_binding["binding_sha256"],
                        "original_probe_sha256": probe_binding[
                            "original_probe_sha256"
                        ],
                        "normalized_probe_sha256": probe_binding[
                            "normalized_probe_sha256"
                        ],
                        "delta": probe_binding["delta"],
                    }
                ),
            },
            "model": {
                "hidden_size": hidden_size,
                "layer_count": layer_count,
                "planned_layers": requested_layers,
                "excluded_prefixes": ["model.visual.", "mtp."],
            },
            "search": cfg.as_dict(),
            "tensor_metadata": tensor_metadata,
            "tensor_metadata_binding": tensor_metadata_binding,
            "scales": {
                "file": PLAN_SCALES,
                "sha256": scales_sha256,
                "vectors": len(scale_tensors),
            },
            "seams": seam_records,
            "transforms": sorted(
                transforms,
                key=lambda row: (
                    str(row["tensor"]), int(row["order"])
                ),
            ),
            "verification": {
                "fp64_invariance_max_abs": worst_invariance,
                "threshold": 1e-10,
                "domain": "pre_cast_fp64_algebra",
                "required_bf16_fold_kl_max": 5e-4,
            },
        }
        if _producer_identity() != producer:
            raise RuntimeError(
                "PrismaSnap producer implementation changed while planning; "
                "refusing an equivocal receipt"
            )
        source.verify_stable()
        plan["plan_sha256"] = _plan_digest(plan)
        _atomic_json(staging / PLAN_JSON, plan)
        _fsync_dir(staging)
        os.replace(staging, output)
        _fsync_dir(output.parent)
        return plan
    except BaseException:
        # Preserve a failed plan directory for diagnosis; its absence of a
        # signed plan.json makes it unusable by materialize/merge.
        raise


def _finite_number(value: object, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{where} must be a finite number")
    return result


def _validate_search_contract(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("PrismaSnap plan search metadata is malformed")
    _require_exact_keys(value, _SEARCH_KEYS, where="PrismaSnap plan search")
    alphas = value.get("alphas")
    levels = value.get("nvfp4_joint_scale_levels")
    variant = value.get("variant")
    integer_fields = ("group_size", "max_rounds", "polish_top", "polish_pool")
    if any(type(value.get(name)) is not int for name in integer_fields):
        raise RuntimeError("PrismaSnap plan search integer fields are malformed")
    if (
        not isinstance(alphas, list)
        or not alphas
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in alphas
        )
        or not isinstance(levels, list)
        or not levels
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in levels
        )
        or not isinstance(variant, list)
        or any(not isinstance(item, str) for item in variant)
        or variant
        != [name for name in ("stage", "polish") if name in set(variant)]
        or type(value.get("nvfp4_snapped_scale_scoring")) is not bool
    ):
        raise RuntimeError("PrismaSnap plan search fields are malformed")
    try:
        config = PrismaSnapSearchConfig(
            group_size=int(value["group_size"]),
            alphas=tuple(float(item) for item in alphas),
            max_rounds=int(value["max_rounds"]),
            stage="stage" in variant,
            polish="polish" in variant,
            polish_top=int(value["polish_top"]),
            polish_pool=int(value["polish_pool"]),
            scale_rule=str(value["nvfp4_scale_rule"]),
            snapped_scale_scoring=bool(
                value["nvfp4_snapped_scale_scoring"]
            ),
            joint_scale_levels=tuple(float(item) for item in levels),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("PrismaSnap plan search contract is invalid") from exc
    canonical = config.as_dict()
    if dict(value) != canonical:
        raise RuntimeError("PrismaSnap plan search contract is noncanonical")
    return canonical


def _validate_tensor_metadata_contract(
    plan: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError("PrismaSnap plan source metadata is malformed")
    identity = source.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("PrismaSnap plan lacks its source identity")
    weight_map = identity.get("checkpoint_weight_map")
    if not isinstance(weight_map, dict) or not weight_map or not all(
        isinstance(key, str)
        and key
        and isinstance(owner, str)
        and owner
        and Path(owner).name == owner
        for key, owner in weight_map.items()
    ):
        raise RuntimeError("PrismaSnap plan source tensor map is malformed")
    metadata = plan.get("tensor_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("PrismaSnap plan lacks full tensor metadata")
    _require_exact_keys(
        metadata, _TENSOR_METADATA_KEYS, where="PrismaSnap tensor metadata"
    )
    tensors = metadata.get("tensors")
    if metadata.get("schema") != TENSOR_METADATA_SCHEMA or not isinstance(
        tensors, dict
    ):
        raise RuntimeError("PrismaSnap tensor metadata schema is malformed")
    unsigned = {"schema": TENSOR_METADATA_SCHEMA, "tensors": tensors}
    if _require_sha256(
        metadata.get("sha256"), where="PrismaSnap tensor metadata.sha256"
    ) != canonical_json_sha256(unsigned, where="PrismaSnap full tensor metadata"):
        raise RuntimeError("PrismaSnap tensor metadata digest mismatch")
    if set(tensors) != set(weight_map):
        raise RuntimeError("PrismaSnap tensor metadata census differs from source")
    result: dict[str, dict[str, object]] = {}
    for name in sorted(tensors):
        raw = tensors[name]
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"PrismaSnap tensor metadata is malformed for {name}")
        _require_exact_keys(
            raw, _TENSOR_ROW_KEYS, where=f"PrismaSnap tensor metadata {name}"
        )
        shape = raw.get("shape")
        dtype = raw.get("dtype")
        owner = raw.get("owner")
        if (
            owner != weight_map[name]
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(dtype, str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]*", dtype) is None
        ):
            raise RuntimeError(f"PrismaSnap tensor metadata is invalid for {name}")
        result[name] = {
            "owner": str(owner),
            "shape": list(shape),
            "dtype": dtype,
        }
    return result


def _require_tensor_layer(name: str, layer: int, *, where: str) -> None:
    matches = [int(value) for value in _SOURCE_BODY_LAYER.findall(name)]
    if matches != [layer]:
        raise RuntimeError(f"{where} is not bound to body layer {layer}: {name}")


def _validate_seam_stats(
    value: object,
    *,
    vector_size: int,
    search: Mapping[str, object],
    where: str,
) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{where} is malformed")
    _require_exact_keys(value, _SEAM_STATS_KEYS, where=where)
    baseline = _finite_number(value.get("error_baseline"), where=f"{where}.error_baseline")
    final = _finite_number(value.get("error_final"), where=f"{where}.error_final")
    improvement = _finite_number(
        value.get("improvement_fraction"), where=f"{where}.improvement_fraction"
    )
    counts = {
        name: _require_nonnegative_int(value.get(name), where=f"{where}.{name}")
        for name in (
            "groups",
            "groups_moved",
            "rounds",
            "candidate_count",
            "polish_pool",
            "polished",
        )
    }
    expected_improvement = 0.0 if baseline == 0.0 else (baseline - final) / baseline
    if (
        value.get("algorithm") != PRISMASNAP_ALGORITHM
        or baseline < 0.0
        or final < 0.0
        or final > baseline
        or not math.isclose(improvement, expected_improvement, rel_tol=1e-12, abs_tol=1e-15)
        or counts["groups"] <= 0
        or counts["groups"] * int(search["group_size"]) != vector_size
        or counts["groups_moved"] > counts["groups"]
        or not (1 <= counts["rounds"] <= int(search["max_rounds"]))
        or counts["candidate_count"] != len(search["alphas"])
        or type(value.get("fell_back")) is not bool
        or counts["polish_pool"] > int(search["polish_pool"])
        or counts["polished"] > counts["polish_pool"]
        or value.get("variant") != search["variant"]
    ):
        raise RuntimeError(f"{where} violates the measured search contract")


def _validate_dense_plan_semantics(
    plan: Mapping[str, object], scales: Mapping[str, torch.Tensor]
) -> None:
    schema = plan.get("schema")
    expected_top = _PLAN_KEYS if schema == PLAN_SCHEMA else _PLAN_SET_KEYS
    _require_exact_keys(plan, expected_top, where="PrismaSnap plan")
    if (
        schema not in {PLAN_SCHEMA, PLAN_SET_SCHEMA}
        or plan.get("state") != "PLANNED"
        or plan.get("algorithm") != PRISMASNAP_ALGORITHM
        or not isinstance(plan.get("producer"), Mapping)
        or not isinstance(plan.get("profile"), str)
        or "dense" not in str(plan.get("profile"))
        or not isinstance(plan.get("probe"), Mapping)
    ):
        raise RuntimeError("PrismaSnap dense plan top-level contract is malformed")

    model = plan.get("model")
    if not isinstance(model, Mapping):
        raise RuntimeError("PrismaSnap plan model metadata is malformed")
    _require_exact_keys(model, _MODEL_KEYS, where="PrismaSnap plan model")
    hidden_size = model.get("hidden_size")
    layer_count = model.get("layer_count")
    planned_layers = model.get("planned_layers")
    if (
        type(hidden_size) is not int
        or hidden_size <= 0
        or type(layer_count) is not int
        or layer_count <= 0
        or not isinstance(planned_layers, list)
        or not planned_layers
        or any(type(layer) is not int for layer in planned_layers)
        or planned_layers != sorted(set(planned_layers))
        or planned_layers[0] < 0
        or planned_layers[-1] >= layer_count
        or model.get("excluded_prefixes") != ["model.visual.", "mtp."]
    ):
        raise RuntimeError("PrismaSnap plan model contract is malformed")

    search = _validate_search_contract(plan.get("search"))
    verification = plan.get("verification")
    if not isinstance(verification, Mapping):
        raise RuntimeError("PrismaSnap plan verification metadata is malformed")
    _require_exact_keys(
        verification, _VERIFICATION_KEYS, where="PrismaSnap plan verification"
    )
    worst = _finite_number(
        verification.get("fp64_invariance_max_abs"),
        where="PrismaSnap plan verification.fp64_invariance_max_abs",
    )
    threshold = _finite_number(
        verification.get("threshold"), where="PrismaSnap plan verification.threshold"
    )
    required_kl = _finite_number(
        verification.get("required_bf16_fold_kl_max"),
        where="PrismaSnap plan verification.required_bf16_fold_kl_max",
    )
    if (
        worst < 0.0
        or threshold != 1e-10
        or worst > threshold
        or verification.get("domain") != "pre_cast_fp64_algebra"
        or required_kl != 5e-4
    ):
        raise RuntimeError("PrismaSnap plan verification contract is invalid")

    tensors = _validate_tensor_metadata_contract(plan)
    tensor_metadata = plan["tensor_metadata"]
    binding = plan.get("tensor_metadata_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("PrismaSnap tensor-metadata binding is malformed")
    _require_exact_keys(
        binding,
        _TENSOR_METADATA_BINDING_KEYS,
        where="PrismaSnap tensor-metadata binding",
    )
    binding_mode = binding.get("mode")
    manifest_sha256 = binding.get("manifest_sha256")
    if (
        binding_mode not in {"inline_full_header_scan", "external_manifest"}
        or binding.get("tensor_metadata_sha256") != tensor_metadata["sha256"]
        or (
            binding_mode == "inline_full_header_scan"
            and manifest_sha256 is not None
        )
        or (
            binding_mode == "external_manifest"
            and _require_sha256(
                manifest_sha256,
                where="PrismaSnap tensor-metadata binding manifest",
            )
            != manifest_sha256
        )
    ):
        raise RuntimeError("PrismaSnap tensor-metadata binding contract is invalid")
    scales_meta = plan.get("scales")
    if not isinstance(scales_meta, Mapping):
        raise RuntimeError("PrismaSnap plan scale metadata is malformed")
    _require_exact_keys(
        scales_meta, _SCALE_METADATA_KEYS, where="PrismaSnap plan scales"
    )
    if (
        scales_meta.get("file") != PLAN_SCALES
        or _require_sha256(
            scales_meta.get("sha256"), where="PrismaSnap plan scales.sha256"
        )
        != scales_meta.get("sha256")
        or type(scales_meta.get("vectors")) is not int
        or scales_meta.get("vectors") != 3 * len(planned_layers)
        or len(scales) != 3 * len(planned_layers)
    ):
        raise RuntimeError("PrismaSnap plan scale contract is malformed")

    seams = plan.get("seams")
    if not isinstance(seams, list) or len(seams) != 3 * len(planned_layers):
        raise RuntimeError("PrismaSnap plan does not have exactly three seams per layer")
    kinds = ("input_norm", "post_attention_norm", "up_down")
    expected_seam_order = [(layer, kind) for layer in planned_layers for kind in kinds]
    actual_seam_order: list[tuple[int, str]] = []
    by_layer: dict[int, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for ordinal, raw in enumerate(seams):
        if not isinstance(raw, Mapping):
            raise RuntimeError("PrismaSnap plan seam is not an object")
        layer = raw.get("layer")
        kind = raw.get("kind")
        if type(layer) is not int or layer not in planned_layers or kind not in kinds:
            raise RuntimeError("PrismaSnap plan seam layer/kind is malformed")
        _require_exact_keys(
            raw,
            _UP_DOWN_SEAM_KEYS if kind == "up_down" else _NORM_SEAM_KEYS,
            where=f"PrismaSnap plan seam[{ordinal}]",
        )
        if kind in by_layer[layer]:
            raise RuntimeError(f"PrismaSnap plan repeats layer {layer} seam {kind}")
        by_layer[layer][str(kind)] = raw
        actual_seam_order.append((layer, str(kind)))
    if actual_seam_order != expected_seam_order:
        raise RuntimeError("PrismaSnap plan seam order/census is noncanonical")

    expected_vectors = {
        f"layer_{layer:05d}_{suffix}"
        for layer in planned_layers
        for suffix in ("input", "post", "updown")
    }
    if set(scales) != expected_vectors:
        raise RuntimeError("PrismaSnap plan scale vectors are not layer-bound")

    expected_transforms: list[dict[str, object]] = []
    for layer in planned_layers:
        rows = by_layer[layer]
        if set(rows) != set(kinds):
            raise RuntimeError(f"PrismaSnap layer {layer} lacks an exact seam triple")
        input_row = rows["input_norm"]
        post_row = rows["post_attention_norm"]
        updown_row = rows["up_down"]
        graph_hashes = {row.get("graph_sha256") for row in rows.values()}
        if len(graph_hashes) != 1:
            raise RuntimeError(f"PrismaSnap layer {layer} seam graphs disagree")
        _require_sha256(
            next(iter(graph_hashes)), where=f"PrismaSnap layer {layer} graph_sha256"
        )
        vector_names = {
            "input_norm": f"layer_{layer:05d}_input",
            "post_attention_norm": f"layer_{layer:05d}_post",
            "up_down": f"layer_{layer:05d}_updown",
        }
        if any(rows[kind].get("vector") != name for kind, name in vector_names.items()):
            raise RuntimeError(f"PrismaSnap layer {layer} seam vector binding is invalid")

        for kind, row in (("input_norm", input_row), ("post_attention_norm", post_row)):
            norm = row.get("norm")
            consumers = row.get("consumers")
            offset = row.get("norm_parameter_offset")
            if (
                not isinstance(norm, str)
                or not isinstance(consumers, list)
                or any(not isinstance(name, str) for name in consumers)
                or len(set(consumers)) != len(consumers)
                or consumers != sorted(consumers, key=_dense_source_role_order)
                or len(consumers) < 2
                or (kind == "post_attention_norm" and len(consumers) != 2)
            ):
                raise RuntimeError(f"PrismaSnap layer {layer} {kind} roles are malformed")
            _finite_number(offset, where=f"PrismaSnap layer {layer} {kind} norm offset")
            for name in [norm, *consumers]:
                if name not in tensors:
                    raise RuntimeError(f"PrismaSnap seam role references absent tensor {name}")
                _require_tensor_layer(name, layer, where="PrismaSnap seam role")
                if tensors[name]["dtype"] != "BF16":
                    raise RuntimeError(f"PrismaSnap seam role is not BF16: {name}")
            vector_size = int(scales[str(row["vector"])].numel())
            norm_shape = tuple(tensors[norm]["shape"])
            if norm_shape != (vector_size,):
                raise RuntimeError(f"PrismaSnap norm/vector shape mismatch: {norm}")
            for name in consumers:
                shape = tuple(tensors[name]["shape"])
                if len(shape) != 2 or shape[1] != vector_size:
                    raise RuntimeError(f"PrismaSnap consumer/vector shape mismatch: {name}")
            _validate_seam_stats(
                row.get("stats"),
                vector_size=vector_size,
                search=search,
                where=f"PrismaSnap layer {layer} {kind} stats",
            )
            expected_transforms.append(
                {
                    "tensor": norm,
                    "vector": row["vector"],
                    "operation": "affine_multiply",
                    "axis": 0,
                    "order": 0,
                    "parameter_offset": offset,
                }
            )
            expected_transforms.extend(
                {
                    "tensor": name,
                    "vector": row["vector"],
                    "operation": "divide",
                    "axis": 1,
                    "order": 0,
                }
                for name in consumers
            )

        gate = updown_row.get("gate")
        up = updown_row.get("up")
        down = updown_row.get("down")
        if not all(isinstance(name, str) for name in (gate, up, down)) or len(
            {gate, up, down}
        ) != 3:
            raise RuntimeError(f"PrismaSnap layer {layer} up/down roles are malformed")
        assert isinstance(gate, str) and isinstance(up, str) and isinstance(down, str)
        post_consumers = post_row["consumers"]
        input_consumers = input_row["consumers"]
        input_offset = float(input_row["norm_parameter_offset"])
        post_offset = float(post_row["norm_parameter_offset"])
        norms = {str(input_row["norm"]), str(post_row["norm"])}
        weight_roles = set(input_consumers) | set(post_consumers) | {down}
        if (
            len(norms) != 2
            or norms & weight_roles
            or set(post_consumers) != {gate, up}
            or set(input_consumers) & {gate, up, down}
        ):
            raise RuntimeError(f"PrismaSnap layer {layer} seam roles are cross-bound")
        if input_offset != post_offset:
            raise RuntimeError(f"PrismaSnap layer {layer} norm encodings disagree")
        executable_graph_sha256 = _dense_plan_graph_sha256(
            layer=layer,
            input_norm=str(input_row["norm"]),
            input_offset=input_offset,
            input_consumers=[str(name) for name in input_consumers],
            post_norm=str(post_row["norm"]),
            post_offset=post_offset,
            post_consumers=[str(name) for name in post_consumers],
            gate=gate,
            up=up,
            down=down,
        )
        if graph_hashes != {executable_graph_sha256}:
            raise RuntimeError(
                f"PrismaSnap layer {layer} graph digest is not role-bound"
            )
        for name in (gate, up, down):
            if name not in tensors:
                raise RuntimeError(f"PrismaSnap seam role references absent tensor {name}")
            _require_tensor_layer(name, layer, where="PrismaSnap seam role")
            if tensors[name]["dtype"] != "BF16":
                raise RuntimeError(f"PrismaSnap seam role is not BF16: {name}")
        gate_shape = tuple(tensors[gate]["shape"])
        up_shape = tuple(tensors[up]["shape"])
        down_shape = tuple(tensors[down]["shape"])
        vector_size = int(scales[str(updown_row["vector"])].numel())
        if (
            len(up_shape) != 2
            or gate_shape != up_shape
            or up_shape[0] != vector_size
            or len(down_shape) != 2
            or down_shape[1] != vector_size
        ):
            raise RuntimeError(f"PrismaSnap layer {layer} up/down shapes are invalid")
        _validate_seam_stats(
            updown_row.get("stats"),
            vector_size=vector_size,
            search=search,
            where=f"PrismaSnap layer {layer} up_down stats",
        )
        expected_transforms.extend(
            [
                {
                    "tensor": up,
                    "vector": updown_row["vector"],
                    "operation": "multiply",
                    "axis": 0,
                    "order": 1,
                },
                {
                    "tensor": down,
                    "vector": updown_row["vector"],
                    "operation": "divide",
                    "axis": 1,
                    "order": 0,
                },
            ]
        )

    raw_transforms = plan.get("transforms")
    expected_transforms.sort(key=lambda row: (str(row["tensor"]), int(row["order"])))
    if not isinstance(raw_transforms, list) or raw_transforms != expected_transforms:
        raise RuntimeError("PrismaSnap plan transform program is not seam-derived")
    for ordinal, raw in enumerate(raw_transforms):
        if not isinstance(raw, Mapping):
            raise RuntimeError("PrismaSnap plan transform is not an object")
        operation = raw.get("operation")
        keys = {"tensor", "vector", "operation", "axis", "order"} | (
            {"parameter_offset"} if operation == "affine_multiply" else set()
        )
        _require_exact_keys(raw, frozenset(keys), where=f"PrismaSnap transform[{ordinal}]")
        if type(raw.get("axis")) is not int or type(raw.get("order")) is not int:
            raise RuntimeError("PrismaSnap transform axis/order must be integers")

    if schema == PLAN_SET_SCHEMA:
        workers = plan.get("workers")
        if not isinstance(workers, list) or len(workers) < 2:
            raise RuntimeError("PrismaSnap merged plan worker census is malformed")
        owned: set[int] = set()
        for ordinal, worker in enumerate(workers):
            if not isinstance(worker, Mapping):
                raise RuntimeError("PrismaSnap merged plan worker is malformed")
            _require_exact_keys(
                worker,
                frozenset({"plan_sha256", "layers"}),
                where=f"PrismaSnap merged plan worker[{ordinal}]",
            )
            _require_sha256(
                worker.get("plan_sha256"),
                where=f"PrismaSnap merged plan worker[{ordinal}].plan_sha256",
            )
            worker_layers = worker.get("layers")
            if (
                not isinstance(worker_layers, list)
                or not worker_layers
                or any(type(layer) is not int for layer in worker_layers)
                or worker_layers != sorted(set(worker_layers))
                or owned & set(worker_layers)
            ):
                raise RuntimeError("PrismaSnap merged plan worker layers are malformed")
            owned.update(worker_layers)
        if owned != set(planned_layers):
            raise RuntimeError("PrismaSnap merged plan workers do not cover planned layers")


def load_plan(plan_dir: str | Path) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    root = Path(plan_dir).resolve(strict=True)
    plan = _load_json(root / PLAN_JSON, where="PrismaSnap plan")
    from . import prismasnap_moe_checkpoint as moe_checkpoint

    dense_schema = plan.get("schema") in {PLAN_SCHEMA, PLAN_SET_SCHEMA}
    moe_schema = plan.get("schema") in {
        moe_checkpoint.MOE_PLAN_SCHEMA,
        moe_checkpoint.MOE_PLAN_SET_SCHEMA,
    }
    if not dense_schema and not moe_schema:
        raise RuntimeError(f"unsupported PrismaSnap plan schema {plan.get('schema')!r}")
    if plan.get("state") != "PLANNED":
        raise RuntimeError("PrismaSnap plan is not in PLANNED state")
    if plan.get("plan_sha256") != _plan_digest(plan):
        raise RuntimeError("PrismaSnap plan digest mismatch")
    scales_meta = plan.get("scales")
    if not isinstance(scales_meta, dict) or scales_meta.get("file") != PLAN_SCALES:
        raise RuntimeError("PrismaSnap plan scale metadata is malformed")
    scale_path = root / PLAN_SCALES
    if _sha256_file(scale_path) != scales_meta.get("sha256"):
        raise RuntimeError("PrismaSnap plan scale content digest mismatch")
    scales = load_file(str(scale_path), device="cpu")
    expected_vectors = (
        {
            str(row["vector"])
            for row in plan.get("seams", [])
            if isinstance(row, dict) and isinstance(row.get("vector"), str)
        }
        if dense_schema
        else moe_checkpoint.plan_scale_vector_names(plan)
    )
    if set(scales) != expected_vectors:
        raise RuntimeError("PrismaSnap plan scale-vector census mismatch")
    for name, value in scales.items():
        allowed_ranks = {1} if dense_schema else {1, 2}
        if value.ndim not in allowed_ranks or value.dtype != torch.float64 or value.numel() == 0:
            raise RuntimeError(
                f"PrismaSnap scale vector {name!r} must be nonempty float64 "
                f"rank {sorted(allowed_ranks)}"
            )
        if not bool(torch.isfinite(value).all().item()) or bool((value <= 0).any().item()):
            raise RuntimeError(
                f"PrismaSnap scale vector {name!r} must be finite and positive"
            )
    if dense_schema:
        _validate_dense_plan_semantics(plan, scales)
    else:
        moe_checkpoint.validate_moe_plan_semantics(plan, scales)
    return plan, scales


def merge_plans(
    plan_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Strictly collate non-overlapping layer plans from multiple workers."""
    if len(plan_dirs) < 2:
        raise ValueError("PrismaSnap plan merge requires at least two worker plans")
    loaded = [load_plan(path) for path in plan_dirs]
    plans = [row[0] for row in loaded]
    from . import prismasnap_moe_checkpoint as moe_checkpoint

    dense_family = all(
        plan.get("schema") in {PLAN_SCHEMA, PLAN_SET_SCHEMA} for plan in plans
    )
    moe_family = all(
        plan.get("schema")
        in {moe_checkpoint.MOE_PLAN_SCHEMA, moe_checkpoint.MOE_PLAN_SET_SCHEMA}
        for plan in plans
    )
    if not dense_family and not moe_family:
        raise RuntimeError("PrismaSnap worker plans mix dense and MoE schemas")
    invariant_fields = (
        "algorithm",
        "producer",
        "profile",
        "search",
        "tensor_metadata",
        "tensor_metadata_binding",
        "promotion",
    )
    for field in invariant_fields:
        if any(plan.get(field) != plans[0].get(field) for plan in plans[1:]):
            raise RuntimeError(f"PrismaSnap worker plans disagree on {field}")
    source0 = plans[0].get("source")
    probe0 = plans[0].get("probe")
    if not isinstance(source0, Mapping) or not isinstance(probe0, Mapping):
        raise RuntimeError("PrismaSnap worker plan has malformed source/probe")
    for plan in plans[1:]:
        source_n = plan.get("source")
        probe_n = plan.get("probe")
        if (
            not isinstance(source_n, Mapping)
            or source_n.get("portable_identity") != source0.get("portable_identity")
            or not isinstance(probe_n, Mapping)
            or {key: value for key, value in probe_n.items() if key != "path"}
            != {key: value for key, value in probe0.items() if key != "path"}
        ):
            raise RuntimeError(
                "PrismaSnap worker plans disagree on portable source/probe content"
            )
    model0 = plans[0].get("model")
    if not isinstance(model0, dict):
        raise RuntimeError("PrismaSnap worker plan has malformed model metadata")
    layer_count = int(model0["layer_count"])
    owners: dict[int, int] = {}
    for worker, plan in enumerate(plans):
        model = plan.get("model")
        if not isinstance(model, dict):
            raise RuntimeError("PrismaSnap worker plan has malformed model metadata")
        per_layer_fields = (
            {"planned_layers", "expert_counts", "routed_layouts"}
            if moe_family
            else {"planned_layers"}
        )
        comparable = {k: v for k, v in model.items() if k not in per_layer_fields}
        baseline = {k: v for k, v in model0.items() if k not in per_layer_fields}
        if comparable != baseline:
            raise RuntimeError("PrismaSnap worker plans disagree on model contract")
        for layer in model.get("planned_layers", []):
            layer = int(layer)
            if layer in owners:
                raise RuntimeError(
                    f"PrismaSnap worker plans overlap on layer {layer}: "
                    f"workers {owners[layer]} and {worker}"
                )
            owners[layer] = worker
    expected = set(range(layer_count))
    if set(owners) != expected:
        missing = sorted(expected - set(owners))
        extra = sorted(set(owners) - expected)
        raise RuntimeError(
            f"PrismaSnap worker plan coverage is not exact; missing={missing[:12]} "
            f"extra={extra[:12]}"
        )
    scales: dict[str, torch.Tensor] = {}
    seams: list[dict[str, object]] = []
    transforms: list[dict[str, object]] = []
    merged_expert_counts: dict[str, object] = {}
    merged_routed_layouts: dict[str, object] = {}
    for plan, worker_scales in loaded:
        overlap = set(scales) & set(worker_scales)
        if overlap:
            raise RuntimeError(f"PrismaSnap scale vectors overlap: {sorted(overlap)[:8]}")
        scales.update(worker_scales)
        seams.extend(plan["seams"])
        transforms.extend(plan["transforms"])
        if moe_family:
            model = plan["model"]
            for field, target in (
                ("expert_counts", merged_expert_counts),
                ("routed_layouts", merged_routed_layouts),
            ):
                raw = model[field]
                if not isinstance(raw, Mapping) or set(target) & set(raw):
                    raise RuntimeError(
                        f"PrismaSnap MoE worker plans overlap on {field}"
                    )
                target.update(raw)
    def merged_payload(scales_sha256: str) -> dict[str, object]:
        merged: dict[str, object] = {
            "schema": (
                PLAN_SET_SCHEMA
                if dense_family
                else moe_checkpoint.MOE_PLAN_SET_SCHEMA
            ),
            "state": "PLANNED",
            "algorithm": plans[0]["algorithm"],
            "producer": plans[0]["producer"],
            "profile": plans[0]["profile"],
            "source": plans[0]["source"],
            "probe": plans[0]["probe"],
            "model": {
                **model0,
                "planned_layers": list(range(layer_count)),
                **(
                    {
                        "expert_counts": dict(sorted(merged_expert_counts.items())),
                        "routed_layouts": dict(sorted(merged_routed_layouts.items())),
                    }
                    if moe_family
                    else {}
                ),
            },
            "search": plans[0]["search"],
            "tensor_metadata": plans[0]["tensor_metadata"],
            "tensor_metadata_binding": plans[0]["tensor_metadata_binding"],
            "scales": {
                "file": PLAN_SCALES,
                "sha256": scales_sha256,
                "vectors": len(scales),
            },
            "seams": sorted(
                seams,
                key=(
                    (lambda row: (int(row["layer"]), str(row["kind"])))
                    if dense_family
                    else moe_checkpoint._seam_sort_key
                ),
            ),
            "transforms": sorted(
                transforms,
                key=lambda row: (str(row["tensor"]), int(row["order"])),
            ),
            "verification": (
                {
                    "fp64_invariance_max_abs": max(
                        float(plan["verification"]["fp64_invariance_max_abs"])
                        for plan in plans
                    ),
                    "threshold": 1e-10,
                    "domain": "pre_cast_fp64_algebra",
                    "required_bf16_fold_kl_max": 5e-4,
                }
                if dense_family
                else {
                    "fp64_invariance_max_abs": max(
                        float(plan["verification"]["fp64_invariance_max_abs"])
                        for plan in plans
                    ),
                    "router_logit_max_abs": max(
                        float(plan["verification"]["router_logit_max_abs"])
                        for plan in plans
                    ),
                    "route_weight_max_abs": max(
                        float(plan["verification"]["route_weight_max_abs"])
                        for plan in plans
                    ),
                    "routed_output_max_abs": max(
                        float(plan["verification"]["routed_output_max_abs"])
                        for plan in plans
                    ),
                    "routing_changed": max(
                        float(plan["verification"]["routing_changed"])
                        for plan in plans
                    ),
                    "threshold": 1e-10,
                    "domain": "pre_cast_fp64_router_routing_and_expert_algebra",
                    "required_bf16_fold_kl_max": 5e-4,
                    "real_moe_fold_kl_evidence": None,
                }
            ),
            "workers": [
                {
                    "plan_sha256": plan["plan_sha256"],
                    "layers": plan["model"]["planned_layers"],
                }
                for plan in plans
            ],
        }
        if moe_family:
            merged["promotion"] = plans[0]["promotion"]
        merged["plan_sha256"] = _plan_digest(merged)
        return merged

    def validate_tree(root: Path) -> dict[str, object]:
        existing, existing_scales = load_plan(root)
        if set(existing_scales) != set(scales) or any(
            not torch.equal(existing_scales[name], scales[name]) for name in scales
        ):
            raise RuntimeError(
                "PrismaSnap merged plan scales differ from ordered worker inputs"
            )
        expected_plan = merged_payload(str(existing["scales"]["sha256"]))
        if existing != expected_plan:
            raise RuntimeError(
                "PrismaSnap merged plan belongs to different ordered worker inputs"
            )
        return existing

    portable = source0.get("portable_identity")
    if not isinstance(portable, Mapping):
        raise RuntimeError("PrismaSnap worker plan lacks portable source identity")
    merge_state = _sealed_state(
        {
            "schema": PLAN_MERGE_STATE_SCHEMA,
            "state": "MERGING",
            "plan_sha256s": [str(plan["plan_sha256"]) for plan in plans],
            "source_portable_content_sha256": _require_sha256(
                portable.get("portable_content_sha256"),
                where="PrismaSnap plan merge portable source identity",
            ),
        }
    )
    output = Path(output_dir)
    staging = output.with_name(output.name + ".prismasnap-plan-incomplete")
    if os.path.lexists(output):
        if not resume or output.is_symlink() or not output.is_dir():
            raise RuntimeError(f"PrismaSnap merged plan output exists: {output}")
        if os.path.lexists(staging):
            raise RuntimeError(
                "PrismaSnap merged plan has both committed and staging trees"
            )
        return validate_tree(output)

    state_path = staging / PLAN_MERGE_STATE_JSON
    if os.path.lexists(staging):
        if not resume or staging.is_symlink() or not staging.is_dir():
            raise RuntimeError(f"stale PrismaSnap merged-plan state exists: {staging}")
        _discard_interrupted_atomic_write(state_path, resume=True)
        _discard_interrupted_atomic_write(staging / PLAN_JSON, resume=True)
        state_exists = os.path.lexists(state_path)
        if state_exists:
            state = _load_json(state_path, where="PrismaSnap plan merge state")
            _validate_sealed_state(
                state,
                merge_state,
                where="PrismaSnap plan merge state",
            )
        if (staging / PLAN_JSON).is_file() and (staging / PLAN_SCALES).is_file():
            merged = validate_tree(staging)
            if state_exists:
                state_path.unlink()
            _fsync_dir(staging)
            os.replace(staging, output)
            _fsync_dir(output.parent)
            return merged
        if os.path.lexists(staging / PLAN_JSON):
            raise RuntimeError("PrismaSnap merged plan has a half-published manifest")
        allowed = {
            PLAN_MERGE_STATE_JSON,
            PLAN_SCALES,
            f".{PLAN_SCALES}.merge.tmp",
        }
        entries = list(staging.iterdir())
        if any(item.name not in allowed or item.is_symlink() for item in entries):
            raise RuntimeError("PrismaSnap plan merge staging contains unexpected files")
        if not state_exists:
            if entries:
                raise RuntimeError("PrismaSnap plan merge staging lacks durable state")
            _atomic_json(state_path, merge_state)
            _fsync_dir(staging)
    else:
        # ``--resume`` is intentionally idempotent even before the first state
        # write, which lets a sealed-stage retry use the exact same argv.
        staging.mkdir(parents=True, exist_ok=False)
        _atomic_json(state_path, merge_state)
        _fsync_dir(staging)

    scale_path = staging / PLAN_SCALES
    scale_temporary = staging / f".{PLAN_SCALES}.merge.tmp"
    if os.path.lexists(scale_temporary):
        if not resume or scale_temporary.is_symlink() or not scale_temporary.is_file():
            raise RuntimeError("unsafe PrismaSnap merged-scale temporary")
        scale_temporary.unlink()
    if os.path.lexists(scale_path):
        if scale_path.is_symlink() or not scale_path.is_file():
            raise RuntimeError("unsafe PrismaSnap merged scale file")
        existing_scales = load_file(str(scale_path), device="cpu")
        if set(existing_scales) != set(scales) or any(
            not torch.equal(existing_scales[name], scales[name]) for name in scales
        ):
            raise RuntimeError("PrismaSnap staged merged scales are equivocal")
    else:
        save_file(
            {name: tensor.contiguous() for name, tensor in sorted(scales.items())},
            str(scale_temporary),
            metadata={"format": "pt", "algorithm": str(plans[0]["algorithm"])},
        )
        with scale_temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(scale_temporary, scale_path)
        _fsync_dir(staging)
    merged = merged_payload(_sha256_file(scale_path))
    _atomic_json(staging / PLAN_JSON, merged)
    if validate_tree(staging) != merged:
        raise RuntimeError("PrismaSnap staged merged plan changed after publication")
    state_path.unlink()
    _fsync_dir(staging)
    os.replace(staging, output)
    _fsync_dir(output.parent)
    return merged


def _validate_materialization_plan(
    plan: Mapping[str, object],
    source: _Checkpoint,
    scales: Mapping[str, torch.Tensor],
) -> None:
    from . import prismasnap_moe_checkpoint as moe_checkpoint

    if plan.get("schema") in {
        moe_checkpoint.MOE_PLAN_SCHEMA,
        moe_checkpoint.MOE_PLAN_SET_SCHEMA,
    }:
        moe_checkpoint.validate_moe_materialization_plan(plan, source, scales)
        return
    model = plan.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("PrismaSnap plan has malformed model metadata")
    expected = list(range(int(model["layer_count"])))
    if model.get("planned_layers") != expected:
        raise RuntimeError(
            "PrismaSnap materialization requires exact full body-layer coverage"
        )
    source_meta = plan.get("source")
    if not isinstance(source_meta, dict):
        raise RuntimeError("PrismaSnap plan lacks source identity")
    identity = validate_streamed_model_identity(
        source_meta.get("identity"), where="PrismaSnap materialization source"
    )
    portable = source_meta.get("portable_identity")
    if (
        not isinstance(portable, Mapping)
        or portable_streamed_model_content_identity(identity) != portable
    ):
        raise RuntimeError(
            "PrismaSnap plan portable source identity does not derive from its "
            "local source identity"
        )
    if identity.get("checkpoint_weight_map") != source.weight_map:
        raise RuntimeError(
            "PrismaSnap materialization source index differs from the plan"
        )
    tensor_metadata = _validate_tensor_metadata_contract(plan)
    for tensor_name, owner in source.weight_map.items():
        if owner not in source.available_shards:
            continue
        shape, dtype = source.metadata(tensor_name)
        planned = tensor_metadata[tensor_name]
        if tuple(planned["shape"]) != shape or planned["dtype"] != dtype:
            raise RuntimeError(
                "PrismaSnap materialization source header differs from the plan: "
                f"{tensor_name}"
            )
    _validate_config_semantics(source.root, identity)
    producer = plan.get("producer")
    if not isinstance(producer, dict) or producer != _producer_identity():
        raise RuntimeError(
            "PrismaSnap materializer implementation/container identity differs "
            "from the planning runtime"
        )
    search = plan.get("search")
    if not isinstance(search, Mapping) or (
        search.get("objective_fold_dtype") != "float32"
        or search.get("global_scale_scope") != "per_tensor"
        or search.get("materialization_rounding")
        != "sequential_bf16_per_transform"
    ):
        raise RuntimeError("PrismaSnap plan does not carry the measured v1 treatment")
    raw_transforms = plan.get("transforms")
    if not isinstance(raw_transforms, list) or not raw_transforms:
        raise RuntimeError("PrismaSnap plan has no transform program")
    per_tensor_orders: dict[str, list[int]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for raw in raw_transforms:
        if not isinstance(raw, dict):
            raise RuntimeError("PrismaSnap plan transform is not an object")
        operation = raw.get("operation")
        expected_keys = {
            "tensor", "vector", "operation", "axis", "order"
        } | ({"parameter_offset"} if operation == "affine_multiply" else set())
        if set(raw) != expected_keys:
            raise RuntimeError("PrismaSnap plan transform fields are malformed")
        tensor_name = raw.get("tensor")
        vector_name = raw.get("vector")
        if not isinstance(tensor_name, str) or tensor_name not in source.weight_map:
            raise RuntimeError("PrismaSnap plan transform references an unknown tensor")
        if not isinstance(vector_name, str) or vector_name not in scales:
            raise RuntimeError("PrismaSnap plan transform references an unknown vector")
        if operation not in {"multiply", "divide", "affine_multiply"}:
            raise RuntimeError("PrismaSnap plan transform operation is unsupported")
        axis = raw.get("axis")
        order = raw.get("order")
        if type(axis) is not int or axis not in {0, 1}:
            raise RuntimeError("PrismaSnap plan transform axis is malformed")
        if type(order) is not int or order < 0:
            raise RuntimeError("PrismaSnap plan transform order is malformed")
        identity = (tensor_name, order)
        if identity in seen:
            raise RuntimeError("PrismaSnap plan repeats a tensor transform order")
        seen.add(identity)
        per_tensor_orders[tensor_name].append(order)
        if operation == "affine_multiply":
            offset = raw.get("parameter_offset")
            if not isinstance(offset, (int, float)) or not math.isfinite(float(offset)):
                raise RuntimeError("PrismaSnap affine transform offset is malformed")
        shard = source.weight_map[tensor_name]
        if shard in source.available_shards:
            shape, dtype = source.metadata(tensor_name)
            if dtype != "BF16":
                raise RuntimeError(
                    f"PrismaSnap transform source must be BF16: {tensor_name}"
                )
            if axis >= len(shape) or shape[axis] != int(scales[vector_name].numel()):
                raise RuntimeError(
                    f"PrismaSnap transform/vector shape mismatch: {tensor_name}"
                )
            if operation == "affine_multiply" and len(shape) != 1:
                raise RuntimeError("PrismaSnap affine norm transform must be rank 1")
            if operation != "affine_multiply" and len(shape) != 2:
                raise RuntimeError("PrismaSnap weight transform must be rank 2")
    for tensor_name, orders in per_tensor_orders.items():
        if sorted(orders) != list(range(len(orders))):
            raise RuntimeError(
                f"PrismaSnap transform order is not contiguous for {tensor_name}"
            )


def _verify_output_census(
    root: Path,
    source: _Checkpoint,
    *,
    compare_source_metadata: bool = True,
) -> dict[str, object]:
    index = _load_json(root / "model.safetensors.index.json", where="output index")
    if index.get("weight_map") != source.weight_map:
        raise RuntimeError("PrismaSnap output index differs from source tensor map")
    shard_entries = [path for path in root.iterdir() if path.suffix == ".safetensors"]
    unsafe = [path.name for path in shard_entries if path.is_symlink() or not path.is_file()]
    if unsafe:
        raise RuntimeError(
            f"PrismaSnap output has unsafe safetensors entries {sorted(unsafe)[:12]}"
        )
    actual_shards = {path.name for path in shard_entries}
    if actual_shards != set(source.shards):
        raise RuntimeError(
            "PrismaSnap output shard-file census differs from its index; "
            f"missing={sorted(set(source.shards) - actual_shards)[:12]} "
            f"extra={sorted(actual_shards - set(source.shards))[:12]}"
        )
    seen: set[str] = set()
    for shard in source.shards:
        with safe_open(str(root / shard), framework="pt") as handle:
            for key in handle.keys():
                if key in seen:
                    raise RuntimeError(f"PrismaSnap output duplicates tensor {key}")
                seen.add(key)
                shape = tuple(map(int, handle.get_slice(key).get_shape()))
                dtype = str(handle.get_slice(key).get_dtype())
                if compare_source_metadata and (shape, dtype) != source.metadata(key):
                    raise RuntimeError(
                        f"PrismaSnap output changed shape/dtype for {key}: {shape}/{dtype}"
                    )
    if seen != set(source.weight_map):
        raise RuntimeError("PrismaSnap output tensor census is incomplete")
    return {
        "tensors": len(seen),
        "shards": len(source.shards),
        "checkpoint_weight_map_sha256": canonical_json_sha256(
            source.weight_map, where="PrismaSnap output checkpoint weight map"
        ),
        "index_sha256": _sha256_file(root / "model.safetensors.index.json"),
    }


def _source_shard_identity(plan: Mapping[str, object]) -> dict[str, dict[str, object]]:
    rows = plan["source"]["identity"]["shards"]
    result = {Path(str(row["path"])).name: dict(row) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("PrismaSnap source identity repeats a shard basename")
    return result


def _transforms_by_tensor(
    plan: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = defaultdict(list)
    for raw in plan["transforms"]:
        row = dict(raw)
        result[str(row["tensor"])].append(row)
    for rows in result.values():
        rows.sort(
            key=lambda row: (
                int(row["order"]), str(row["operation"]), str(row["vector"])
            )
        )
    return result


def _discard_interrupted_atomic_write(path: Path, *, resume: bool) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if not os.path.lexists(temporary):
        return
    if not resume or temporary.is_symlink() or not temporary.is_file():
        raise RuntimeError(f"unsafe/stale PrismaSnap atomic temporary: {temporary}")
    temporary.unlink()


def _materialize_selected_shards(
    source: _Checkpoint,
    plan: Mapping[str, object],
    scales: Mapping[str, torch.Tensor],
    temporary: Path,
    selected_shards: Sequence[str],
    *,
    execution_device: torch.device,
    resume: bool,
) -> tuple[list[dict[str, object]], int]:
    if os.path.lexists(temporary):
        if not resume or temporary.is_symlink() or not temporary.is_dir():
            raise RuntimeError(
                f"PrismaSnap incomplete output exists at {temporary}; pass --resume "
                "only for the same content-bound plan"
            )
        state_path = temporary / "materialization_state.json"
        _discard_interrupted_atomic_write(state_path, resume=True)
        if not state_path.exists():
            if any(temporary.iterdir()):
                raise RuntimeError(
                    "PrismaSnap incomplete output has no materialization state"
                )
            _atomic_json(
                state_path,
                {
                    "schema": "prismaquant.prismasnap.materialization_state.v1",
                    "state": "MATERIALIZING",
                    "source_content_sha256": plan["source"]["identity"]["content_sha256"],
                    "plan_sha256": plan["plan_sha256"],
                    "scales_sha256": plan["scales"]["sha256"],
                    "shards": list(selected_shards),
                },
            )
    else:
        # ``--resume`` means ensure-this-output, not resume-only.  Campaign
        # stages therefore use one immutable argv for their first attempt and
        # every bounded retry.
        temporary.mkdir(parents=True, exist_ok=False)
        _atomic_json(
            temporary / "materialization_state.json",
            {
                "schema": "prismaquant.prismasnap.materialization_state.v1",
                "state": "MATERIALIZING",
                "source_content_sha256": plan["source"]["identity"]["content_sha256"],
                "plan_sha256": plan["plan_sha256"],
                "scales_sha256": plan["scales"]["sha256"],
                "shards": list(selected_shards),
            },
        )
    state = _load_json(
        temporary / "materialization_state.json", where="materialization state"
    )
    if (
        state.get("plan_sha256") != plan["plan_sha256"]
        or state.get("source_content_sha256")
        != plan["source"]["identity"]["content_sha256"]
        or state.get("scales_sha256") != plan["scales"]["sha256"]
        or state.get("shards") != list(selected_shards)
    ):
        raise RuntimeError("PrismaSnap resume state belongs to different inputs")

    by_tensor = _transforms_by_tensor(plan)
    source_id = _source_shard_identity(plan)
    receipts_dir = temporary / ".prismasnap-receipts"
    if os.path.lexists(receipts_dir):
        if receipts_dir.is_symlink() or not receipts_dir.is_dir():
            raise RuntimeError(f"unsafe PrismaSnap receipt directory: {receipts_dir}")
    else:
        receipts_dir.mkdir()
        _fsync_dir(temporary)
    receipts: list[dict[str, object]] = []
    changed_total = 0
    for ordinal, shard in enumerate(selected_shards, start=1):
        source_path = source.root / shard
        target_path = temporary / shard
        receipt_path = receipts_dir / f"{shard}.json"
        _discard_interrupted_atomic_write(receipt_path, resume=resume)
        tmp_shard = temporary / f".{shard}.tmp"
        if os.path.lexists(tmp_shard):
            if not resume or tmp_shard.is_symlink() or not tmp_shard.is_file():
                raise RuntimeError(f"unsafe/stale PrismaSnap shard temp: {tmp_shard}")
            tmp_shard.unlink()
        target_exists = os.path.lexists(target_path)
        receipt_exists = os.path.lexists(receipt_path)
        if target_exists and (target_path.is_symlink() or not target_path.is_file()):
            raise RuntimeError(f"unsafe PrismaSnap output shard state: {target_path}")
        if receipt_exists and (receipt_path.is_symlink() or not receipt_path.is_file()):
            raise RuntimeError(f"unsafe PrismaSnap shard receipt state: {receipt_path}")
        if receipt_exists and target_exists:
            receipt = _load_json(receipt_path, where="PrismaSnap shard receipt")
            if (
                receipt.get("schema") != SHARD_RECEIPT_SCHEMA
                or receipt.get("plan_sha256") != plan["plan_sha256"]
                or receipt.get("output_sha256") != _sha256_file(target_path)
                or receipt.get("output_bytes") != target_path.stat().st_size
                or receipt.get("source_sha256") != source_id[shard]["sha256"]
            ):
                raise RuntimeError(f"PrismaSnap resume shard receipt failed: {shard}")
            receipts.append(receipt)
            changed_total += int(receipt.get("changed_tensors", 0))
            print(f"[prismasnap-write] resume verified {shard}", flush=True)
            continue
        if target_exists or receipt_exists:
            if not resume:
                raise RuntimeError(
                    f"PrismaSnap found half-published shard state for {shard}; refusing"
                )
            # An atomic shard rename and its small receipt cannot be one
            # filesystem transaction.  On explicit resume, discard only the
            # uncommitted half-pair inside this plan-bound staging directory
            # and deterministically render that shard again.
            if target_exists:
                target_path.unlink()
            if receipt_exists:
                receipt_path.unlink()
        print(
            f"[prismasnap-write] shard {ordinal}/{len(selected_shards)} {shard}",
            flush=True,
        )
        source_fingerprint = source._fingerprint(source_path)
        actual_source_sha256 = _sha256_file(source_path)
        if actual_source_sha256 != source_id[shard]["sha256"]:
            raise RuntimeError(f"PrismaSnap source shard content changed: {source_path}")
        source.record_verified_shard(shard, source_fingerprint)
        tensors: dict[str, torch.Tensor] = {}
        changed_here = 0
        with safe_open(str(source_path), framework="pt") as handle:
            for key in sorted(handle.keys()):
                value = handle.get_tensor(key)
                transforms = by_tensor.get(key, [])
                if transforms:
                    if str(handle.get_slice(key).get_dtype()) != "BF16":
                        raise RuntimeError(
                            f"PrismaSnap refuses to refold non-BF16 tensor {key}"
                        )
                    source_dtype = value.dtype
                    value = value.to(execution_device, non_blocking=True)
                    for transform in transforms:
                        vector = scales.get(str(transform["vector"]))
                        if vector is None:
                            raise RuntimeError(
                                f"PrismaSnap transform references absent vector "
                                f"{transform['vector']}"
                            )
                        from . import prismasnap_moe_checkpoint as moe_checkpoint

                        if plan.get("schema") in {
                            moe_checkpoint.MOE_PLAN_SCHEMA,
                            moe_checkpoint.MOE_PLAN_SET_SCHEMA,
                        }:
                            value = moe_checkpoint.apply_moe_materialization_transform(
                                value,
                                vector,
                                transform,
                                output_dtype=source_dtype,
                            )
                        else:
                            value = apply_diagonal_transform(
                                value,
                                vector,
                                str(transform["operation"]),  # type: ignore[arg-type]
                                int(transform["axis"]),
                                parameter_offset=float(
                                    transform.get("parameter_offset", 0.0)
                                ),
                                # Release-v1 is the measured prototype contract:
                                # each logical fold is rounded back to source BF16
                                # before the next fold on the same tensor.
                                output_dtype=source_dtype,
                            )
                    value = value.cpu()
                    changed_here += 1
                tensors[key] = value.contiguous()
        source._assert_shard_stable(shard)
        expected_keys = {
            key for key, owner in source.weight_map.items() if owner == shard
        }
        if set(tensors) != expected_keys:
            raise RuntimeError(f"PrismaSnap source shard tensor census changed: {shard}")
        save_file(tensors, str(tmp_shard), metadata={"format": "pt"})
        with tmp_shard.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_shard, target_path)
        _fsync_dir(temporary)
        receipt = {
            "schema": SHARD_RECEIPT_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "source_name": shard,
            "source_bytes": source_path.stat().st_size,
            "source_sha256": actual_source_sha256,
            "output_bytes": target_path.stat().st_size,
            "output_sha256": _sha256_file(target_path),
            "tensor_count": len(tensors),
            "changed_tensors": changed_here,
        }
        _atomic_json(receipt_path, receipt)
        receipts.append(receipt)
        changed_total += changed_here
        del tensors
    return receipts, changed_total


def _copy_checkpoint_metadata(source: _Checkpoint, destination: Path) -> None:
    skip = set(source.shards) | {"model.safetensors.index.json", PROVENANCE_JSON}
    for path in sorted(source.root.iterdir(), key=lambda item: item.name):
        if path.name in skip or path.name.startswith(".") or not path.is_file():
            continue
        if path.suffix == ".safetensors":
            raise RuntimeError(
                "PrismaSnap source contains an unindexed safetensors payload; "
                f"refusing to copy ambiguous checkpoint data: {path}"
            )
        target = destination / path.name
        _copy_file_durable(path, target)
    _copy_file_durable(
        source.root / "model.safetensors.index.json",
        destination / "model.safetensors.index.json",
    )


def _shard_content_identity(receipts: Sequence[Mapping[str, object]]) -> str:
    return canonical_json_sha256(
        [
            {
                "name": str(row["source_name"]),
                "size": int(row["output_bytes"]),
                "sha256": str(row["output_sha256"]),
            }
            for row in sorted(receipts, key=lambda row: str(row["source_name"]))
        ],
        where="PrismaSnap output shard identity",
    )


def _build_provenance(
    plan: Mapping[str, object],
    *,
    census: Mapping[str, object],
    receipts: Sequence[Mapping[str, object]],
    changed_tensors: int,
) -> dict[str, object]:
    by_tensor = _transforms_by_tensor(plan)
    if changed_tensors != len(by_tensor):
        raise RuntimeError(
            "PrismaSnap materialization transformed-tensor census mismatch: "
            f"{changed_tensors} != {len(by_tensor)}"
        )
    from . import prismasnap_moe_checkpoint as moe_checkpoint

    is_moe = plan.get("schema") in {
        moe_checkpoint.MOE_PLAN_SCHEMA,
        moe_checkpoint.MOE_PLAN_SET_SCHEMA,
    }

    def summary(row: Mapping[str, object]) -> dict[str, object]:
        raw_stats = row["stats"]
        stats_rows = raw_stats if isinstance(raw_stats, list) else [raw_stats]
        if not stats_rows or any(not isinstance(value, Mapping) for value in stats_rows):
            raise RuntimeError("PrismaSnap provenance seam stats are malformed")
        groups = sum(int(value["groups"]) for value in stats_rows)
        moved = sum(int(value["groups_moved"]) for value in stats_rows)
        baseline = sum(float(value["error_baseline"]) for value in stats_rows)
        final = sum(float(value["error_final"]) for value in stats_rows)
        return {
            "layer": int(row["layer"]),
            "kind": str(row["kind"]),
            "graph_sha256": str(row["graph_sha256"]),
            "groups": groups,
            "groups_moved": moved,
            "improvement_fraction": (
                0.0 if baseline == 0.0 else (baseline - final) / baseline
            ),
        }

    seam_summary = [summary(row) for row in plan["seams"]]
    provenance: dict[str, object] = {
        "schema": (
            moe_checkpoint.MOE_PROVENANCE_SCHEMA if is_moe else PROVENANCE_SCHEMA
        ),
        # Algebra/census verification is complete here.  Numerical fold
        # fidelity is a separate served-BF16 gate; only its attestor may
        # transition this receipt to VERIFIED.
        "state": "MATERIALIZED",
        "algorithm": (
            moe_checkpoint.PRISMASNAP_MOE_ALGORITHM
            if is_moe
            else PRISMASNAP_ALGORITHM
        ),
        "purely_additive_source_preparation": True,
        "serve_time_changes": False,
        "source_portable_content_sha256": plan["source"]["portable_identity"][
            "portable_content_sha256"
        ],
        "source_local_content_sha256": plan["source"]["identity"]["content_sha256"],
        "source_model": plan["source"]["identity"]["source"],
        "probe_sha256": plan["probe"]["sha256"],
        "calibration": {
            key: plan["probe"].get(key)
            for key in (
                "calib_hash", "dataset", "nsamples", "seqlen",
                "calibration_modality",
            )
        },
        "plan_sha256": plan["plan_sha256"],
        "scales_sha256": plan["scales"]["sha256"],
        "producer": plan["producer"],
        "search": plan["search"],
        "coverage": {
            "body_layers": plan["model"]["planned_layers"],
            "excluded_prefixes": plan["model"]["excluded_prefixes"],
            "seams": len(plan["seams"]),
            "transformed_tensors": len(by_tensor),
            "materialized_changed_tensors": changed_tensors,
        },
        "fp64_invariance": plan["verification"],
        "seam_summary": seam_summary,
        "output": {
            **dict(census),
            "shard_content_sha256": _shard_content_identity(receipts),
        },
    }
    if is_moe:
        provenance["promotion"] = moe_checkpoint.PRISMASNAP_MOE_PROMOTION
        provenance["real_moe_fold_kl_evidence"] = None
    provenance["provenance_sha256"] = canonical_json_sha256(
        provenance, where="PrismaSnap provenance"
    )
    return provenance


def _validate_provenance_digest(
    payload: Mapping[str, object], *, where: str
) -> None:
    claimed = payload.get("provenance_sha256")
    unsigned = {
        str(key): value
        for key, value in payload.items()
        if key != "provenance_sha256"
    }
    from . import prismasnap_moe_checkpoint as moe_checkpoint

    dense = payload.get("schema") == PROVENANCE_SCHEMA
    moe = payload.get("schema") == moe_checkpoint.MOE_PROVENANCE_SCHEMA
    valid_state = (
        payload.get("state") in {"MATERIALIZED", "VERIFIED"}
        if dense
        else payload.get("state") == "MATERIALIZED"
    )
    if (
        not (dense or moe)
        or not valid_state
        or (
            moe
            and (
                payload.get("algorithm") != moe_checkpoint.PRISMASNAP_MOE_ALGORITHM
                or payload.get("promotion")
                != moe_checkpoint.PRISMASNAP_MOE_PROMOTION
                or payload.get("real_moe_fold_kl_evidence") is not None
            )
        )
        or claimed != canonical_json_sha256(unsigned, where=where)
    ):
        raise RuntimeError(f"{where} contract failed")


def _validate_materialized_checkpoint_tree(
    root: Path,
    *,
    source: _Checkpoint,
    plan: Mapping[str, object],
    compare_source_metadata: bool = True,
) -> dict[str, object]:
    provenance = _load_json(
        root / PROVENANCE_JSON, where="PrismaSnap committed provenance"
    )
    _validate_provenance_digest(
        provenance, where="PrismaSnap committed provenance"
    )
    if (
        provenance.get("plan_sha256") != plan["plan_sha256"]
        or provenance.get("scales_sha256") != plan["scales"]["sha256"]
        or provenance.get("producer") != plan["producer"]
    ):
        raise RuntimeError("PrismaSnap committed provenance belongs to other inputs")
    _validate_config_semantics(root, plan["source"]["identity"])
    census = _verify_output_census(
        root,
        source,
        compare_source_metadata=compare_source_metadata,
    )
    output_meta = provenance.get("output")
    if not isinstance(output_meta, Mapping) or any(
        output_meta.get(key) != value for key, value in census.items()
    ):
        raise RuntimeError("PrismaSnap committed output census changed")
    reconstructed: list[dict[str, object]] = []
    for name in source.shards:
        path = root / name
        reconstructed.append(
            {
                "source_name": name,
                "output_bytes": path.stat().st_size,
                "output_sha256": _sha256_file(path),
            }
        )
    if output_meta.get("shard_content_sha256") != _shard_content_identity(
        reconstructed
    ):
        raise RuntimeError("PrismaSnap committed shard content changed")
    return provenance


def _resume_commit_ready_checkpoint(
    temporary: Path,
    output: Path,
    *,
    source: _Checkpoint,
    plan: Mapping[str, object],
    resume: bool,
) -> dict[str, object] | None:
    """Commit a fully verified staging tree stranded after state cleanup."""
    state_path = temporary / "materialization_state.json"
    if not os.path.lexists(temporary) or os.path.lexists(state_path):
        return None
    if not resume or temporary.is_symlink() or not temporary.is_dir():
        return None
    provenance = _validate_materialized_checkpoint_tree(
        temporary, source=source, plan=plan
    )
    receipts_dir = temporary / ".prismasnap-receipts"
    if os.path.lexists(receipts_dir):
        if receipts_dir.is_symlink() or not receipts_dir.is_dir():
            raise RuntimeError("unsafe PrismaSnap commit-ready receipt directory")
        shutil.rmtree(receipts_dir)
    _fsync_dir(temporary)
    os.replace(temporary, output)
    _fsync_dir(output.parent)
    return provenance


def materialize_checkpoint(
    source_dir: str | Path,
    plan_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    resume: bool = False,
    production: bool = False,
) -> dict[str, object]:
    """Stream a plan into a BF16 checkpoint with atomic shard-level resume."""
    if production:
        _require_production_execution(device)
    source = _Checkpoint(Path(source_dir))
    source_provenance = source.root / PROVENANCE_JSON
    if os.path.lexists(source_provenance):
        if source_provenance.is_symlink() or not source_provenance.is_file():
            raise RuntimeError("PrismaSnap source has an unsafe provenance marker")
        raise RuntimeError(
            "PrismaSnap refuses an already-snapped source checkpoint; "
            "double application is outside the production contract"
        )
    plan, scales = load_plan(plan_dir)
    _validate_materialization_plan(plan, source, scales)
    execution_device = torch.device(device)
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PrismaSnap production materialization requires CUDA")
    output = Path(output_dir)
    if os.path.lexists(output):
        if resume and not output.is_symlink() and output.is_dir():
            return _validate_materialized_checkpoint_tree(
                output, source=source, plan=plan
            )
        raise RuntimeError(f"PrismaSnap output already exists: {output}")
    temporary = output.with_name(output.name + ".prismasnap-incomplete")
    recovered = _resume_commit_ready_checkpoint(
        temporary,
        output,
        source=source,
        plan=plan,
        resume=resume,
    )
    if recovered is not None:
        return recovered
    shard_receipts, changed_tensors = _materialize_selected_shards(
        source,
        plan,
        scales,
        temporary,
        source.shards,
        execution_device=execution_device,
        resume=resume,
    )
    _copy_checkpoint_metadata(source, temporary)
    _validate_config_semantics(temporary, plan["source"]["identity"])
    census = _verify_output_census(temporary, source)
    provenance = _build_provenance(
        plan,
        census=census,
        receipts=shard_receipts,
        changed_tensors=changed_tensors,
    )
    _discard_interrupted_atomic_write(temporary / PROVENANCE_JSON, resume=resume)
    _atomic_json(temporary / PROVENANCE_JSON, provenance)
    (temporary / "materialization_state.json").unlink()
    shutil.rmtree(temporary / ".prismasnap-receipts")
    _fsync_dir(temporary)
    os.replace(temporary, output)
    _fsync_dir(output.parent)
    return provenance


def materialize_checkpoint_part(
    source_dir: str | Path,
    plan_dir: str | Path,
    output_dir: str | Path,
    shards: Sequence[str],
    *,
    device: str = "cuda",
    resume: bool = False,
    production: bool = False,
) -> dict[str, object]:
    """Materialize a content-bound subset of original HF shards on a worker."""
    if production:
        _require_production_execution(device)
    source = _Checkpoint(Path(source_dir), require_all_shards=False)
    source_provenance = source.root / PROVENANCE_JSON
    if os.path.lexists(source_provenance):
        if source_provenance.is_symlink() or not source_provenance.is_file():
            raise RuntimeError("PrismaSnap source has an unsafe provenance marker")
        raise RuntimeError("PrismaSnap refuses an already-snapped source")
    plan, scales = load_plan(plan_dir)
    _validate_materialization_plan(plan, source, scales)
    selected = sorted(set(str(name) for name in shards))
    if not selected or not set(selected) < set(source.shards):
        raise ValueError(
            "PrismaSnap checkpoint part must be a nonempty proper shard subset"
        )
    unknown = set(selected) - set(source.shards)
    if unknown:
        raise ValueError(f"PrismaSnap checkpoint part has unknown shards {sorted(unknown)}")
    source.require_shards(selected, where="PrismaSnap checkpoint part")
    execution_device = torch.device(device)
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PrismaSnap production materialization requires CUDA")
    output = Path(output_dir)
    if os.path.lexists(output):
        if resume and not output.is_symlink() and output.is_dir():
            manifest = _load_part(output, plan=plan, source=source)
            listed = sorted(
                str(row["source_name"]) for row in manifest["shards"]
            )
            if listed != selected:
                raise RuntimeError("PrismaSnap committed part has different shards")
            return manifest
        raise RuntimeError(f"PrismaSnap checkpoint-part output exists: {output}")
    temporary = output.with_name(output.name + ".prismasnap-incomplete")
    if (
        resume
        and os.path.lexists(temporary)
        and not os.path.lexists(temporary / "materialization_state.json")
        and (temporary / "part.json").is_file()
    ):
        manifest = _load_part(
            temporary,
            plan=plan,
            source=source,
            allow_recovery_files=True,
        )
        listed = sorted(str(row["source_name"]) for row in manifest["shards"])
        if listed != selected:
            raise RuntimeError("PrismaSnap commit-ready part has different shards")
        receipts_dir = temporary / ".prismasnap-receipts"
        if os.path.lexists(receipts_dir):
            if receipts_dir.is_symlink() or not receipts_dir.is_dir():
                raise RuntimeError("unsafe PrismaSnap commit-ready part receipts")
            shutil.rmtree(receipts_dir)
        _fsync_dir(temporary)
        os.replace(temporary, output)
        _fsync_dir(output.parent)
        return manifest
    receipts, changed = _materialize_selected_shards(
        source,
        plan,
        scales,
        temporary,
        selected,
        execution_device=execution_device,
        resume=resume,
    )
    manifest: dict[str, object] = {
        "schema": PART_SCHEMA,
        "state": "MATERIALIZED",
        "plan_sha256": plan["plan_sha256"],
        "producer": plan["producer"],
        "plan_source_local_content_sha256": plan["source"]["identity"]["content_sha256"],
        "source_portable_content_sha256": plan["source"]["portable_identity"][
            "portable_content_sha256"
        ],
        "shards": receipts,
        "changed_tensors": changed,
    }
    manifest["part_sha256"] = canonical_json_sha256(
        manifest, where="PrismaSnap checkpoint part"
    )
    _discard_interrupted_atomic_write(temporary / "part.json", resume=resume)
    _atomic_json(temporary / "part.json", manifest)
    (temporary / "materialization_state.json").unlink()
    shutil.rmtree(temporary / ".prismasnap-receipts")
    if _load_part(temporary, plan=plan, source=source) != manifest:
        raise RuntimeError("PrismaSnap checkpoint part changed before commit")
    _fsync_dir(temporary)
    os.replace(temporary, output)
    _fsync_dir(output.parent)
    return manifest


def _load_part(
    path: Path,
    *,
    plan: Mapping[str, object],
    source: _Checkpoint,
    allow_recovery_files: bool = False,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"PrismaSnap checkpoint part is not a real directory: {path}")
    payload = _load_json(path / "part.json", where="PrismaSnap checkpoint part")
    _require_exact_keys(payload, _PART_KEYS, where="PrismaSnap checkpoint part")
    claimed = payload.get("part_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "part_sha256"}
    if (
        payload.get("schema") != PART_SCHEMA
        or payload.get("state") != "MATERIALIZED"
        or _require_sha256(
            claimed, where="PrismaSnap checkpoint part.part_sha256"
        )
        != canonical_json_sha256(unsigned, where="PrismaSnap checkpoint part")
        or payload.get("plan_sha256") != plan["plan_sha256"]
        or payload.get("producer") != plan["producer"]
        or payload.get("plan_source_local_content_sha256")
        != plan["source"]["identity"]["content_sha256"]
        or payload.get("source_portable_content_sha256")
        != plan["source"]["portable_identity"]["portable_content_sha256"]
    ):
        raise RuntimeError(f"PrismaSnap checkpoint part contract failed: {path}")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise RuntimeError(f"PrismaSnap checkpoint part has no shard receipts: {path}")
    source_identity = _source_shard_identity(plan)
    transformed = set(_transforms_by_tensor(plan))
    tensor_metadata = _validate_tensor_metadata_contract(plan)
    listed: set[str] = set()
    changed_total = 0
    ordered_names: list[str] = []
    for ordinal, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"PrismaSnap checkpoint part has malformed shard: {path}")
        where = f"PrismaSnap checkpoint part shard[{ordinal}]"
        _require_exact_keys(raw, _SHARD_RECEIPT_KEYS, where=where)
        name = raw.get("source_name")
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError(f"{where}.source_name is malformed")
        if name in listed or name not in source.shards:
            raise RuntimeError(f"PrismaSnap checkpoint part shard census failed: {path}")
        listed.add(name)
        ordered_names.append(name)
        expected_keys = {
            key for key, owner in source.weight_map.items() if owner == name
        }
        expected_changed = len(expected_keys & transformed)
        source_row = source_identity[name]
        source_bytes = _require_nonnegative_int(
            raw.get("source_bytes"), where=f"{where}.source_bytes"
        )
        output_bytes = _require_nonnegative_int(
            raw.get("output_bytes"), where=f"{where}.output_bytes"
        )
        tensor_count = _require_nonnegative_int(
            raw.get("tensor_count"), where=f"{where}.tensor_count"
        )
        changed = _require_nonnegative_int(
            raw.get("changed_tensors"), where=f"{where}.changed_tensors"
        )
        if (
            raw.get("schema") != SHARD_RECEIPT_SCHEMA
            or raw.get("plan_sha256") != plan["plan_sha256"]
            or source_bytes != int(source_row["size"])
            or _require_sha256(
                raw.get("source_sha256"), where=f"{where}.source_sha256"
            )
            != source_row["sha256"]
            or output_bytes <= 0
            or tensor_count != len(expected_keys)
            or changed != expected_changed
        ):
            raise RuntimeError(f"PrismaSnap checkpoint part receipt failed: {name}")
        output_sha256 = _require_sha256(
            raw.get("output_sha256"), where=f"{where}.output_sha256"
        )
        shard_path = path / name
        if (
            shard_path.is_symlink()
            or not shard_path.is_file()
            or shard_path.stat().st_size != output_bytes
            or _sha256_file(shard_path) != output_sha256
        ):
            raise RuntimeError(f"PrismaSnap checkpoint part shard digest failed: {shard_path}")
        with safe_open(str(shard_path), framework="pt") as handle:
            keys = set(handle.keys())
            for key in keys:
                shape = tuple(map(int, handle.get_slice(key).get_shape()))
                dtype = str(handle.get_slice(key).get_dtype())
                planned = tensor_metadata.get(key)
                if (
                    planned is None
                    or planned["owner"] != name
                    or tuple(planned["shape"]) != shape
                    or planned["dtype"] != dtype
                ):
                    raise RuntimeError(
                        "PrismaSnap checkpoint part changed tensor metadata: "
                        f"{key}"
                    )
        if keys != expected_keys:
            raise RuntimeError(f"PrismaSnap checkpoint part tensor census failed: {shard_path}")
        changed_total += changed
    if ordered_names != sorted(ordered_names):
        raise RuntimeError("PrismaSnap checkpoint part shard receipts are not ordered")
    if _require_nonnegative_int(
        payload.get("changed_tensors"),
        where="PrismaSnap checkpoint part.changed_tensors",
    ) != changed_total:
        raise RuntimeError("PrismaSnap checkpoint part changed-tensor total differs")
    expected_entries = {"part.json", *listed}
    if allow_recovery_files:
        expected_entries.add(".prismasnap-receipts")
    actual_entries = {item.name for item in path.iterdir()}
    if actual_entries - expected_entries:
        raise RuntimeError(
            "PrismaSnap checkpoint part has extra files "
            f"{sorted(actual_entries - expected_entries)}"
        )
    recovery = path / ".prismasnap-receipts"
    if recovery.name in actual_entries and (
        recovery.is_symlink() or not recovery.is_dir()
    ):
        raise RuntimeError("unsafe PrismaSnap checkpoint part recovery directory")
    return payload


def _part_collation_binding(
    ordinal: int,
    root: Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "part_sha256": str(payload["part_sha256"]),
        "manifest_file_sha256": _sha256_file(root / "part.json"),
        "shards": [
            {
                "source_name": str(row["source_name"]),
                "source_bytes": int(row["source_bytes"]),
                "source_sha256": str(row["source_sha256"]),
                "output_bytes": int(row["output_bytes"]),
                "output_sha256": str(row["output_sha256"]),
                "tensor_count": int(row["tensor_count"]),
                "changed_tensors": int(row["changed_tensors"]),
                "receipt_sha256": canonical_json_sha256(
                    row, where="PrismaSnap part shard receipt"
                ),
            }
            for row in payload["shards"]
        ],
    }


def _collation_metadata_bindings(source: _Checkpoint) -> list[dict[str, object]]:
    skip = set(source.shards) | {PROVENANCE_JSON}
    paths = [
        path
        for path in sorted(source.root.iterdir(), key=lambda item: item.name)
        if path.name not in skip and not path.name.startswith(".") and path.is_file()
    ]
    bindings: list[dict[str, object]] = []
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"unsafe PrismaSnap collation metadata: {path}")
        bindings.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return bindings


def _collation_stanza(
    part_bindings: Sequence[Mapping[str, object]],
    metadata_bindings: Sequence[Mapping[str, object]],
    transfer_strategy: str,
) -> dict[str, object]:
    normalized_parts = [dict(row) for row in part_bindings]
    normalized_metadata = [dict(row) for row in metadata_bindings]
    return {
        "parts": [str(row["part_sha256"]) for row in normalized_parts],
        "ordered_part_bindings": normalized_parts,
        "ordered_part_bindings_sha256": canonical_json_sha256(
            normalized_parts, where="PrismaSnap ordered part bindings"
        ),
        "source_metadata": normalized_metadata,
        "source_metadata_sha256": canonical_json_sha256(
            normalized_metadata, where="PrismaSnap source metadata bindings"
        ),
        "shard_transfer_strategy": transfer_strategy,
        "exact_disjoint_shard_union": True,
    }


def _build_collated_provenance(
    plan: Mapping[str, object],
    *,
    census: Mapping[str, object],
    receipts: Sequence[Mapping[str, object]],
    changed_tensors: int,
    part_bindings: Sequence[Mapping[str, object]],
    metadata_bindings: Sequence[Mapping[str, object]],
    transfer_strategy: str,
) -> dict[str, object]:
    provenance = _build_provenance(
        plan,
        census=census,
        receipts=receipts,
        changed_tensors=changed_tensors,
    )
    provenance["collation"] = _collation_stanza(
        part_bindings, metadata_bindings, transfer_strategy
    )
    provenance.pop("provenance_sha256", None)
    provenance["provenance_sha256"] = canonical_json_sha256(
        provenance, where="PrismaSnap collated provenance"
    )
    return provenance


def _validate_collated_checkpoint_tree(
    root: Path,
    *,
    source: _Checkpoint,
    plan: Mapping[str, object],
    receipts: Sequence[Mapping[str, object]],
    changed_tensors: int,
    part_bindings: Sequence[Mapping[str, object]],
    metadata_bindings: Sequence[Mapping[str, object]],
    transfer_strategy: str,
    allow_recovery_files: bool = False,
) -> dict[str, object]:
    expected_entries = {
        PROVENANCE_JSON,
        *(str(row["name"]) for row in metadata_bindings),
        *source.shards,
    }
    actual_entries = {item.name for item in root.iterdir()}
    allowed_extra = (
        {PART_MERGE_STATE_JSON, PART_MERGE_RECEIPTS_DIR}
        if allow_recovery_files
        else set()
    )
    if (
        expected_entries - actual_entries
        or actual_entries - expected_entries - allowed_extra
    ):
        raise RuntimeError(
            "PrismaSnap collated checkpoint file census differs: "
            f"missing={sorted(expected_entries - actual_entries)} "
            f"extra={sorted(actual_entries - expected_entries)}"
        )
    provenance = _validate_materialized_checkpoint_tree(
        root,
        source=source,
        plan=plan,
        compare_source_metadata=False,
    )
    for row in metadata_bindings:
        path = root / str(row["name"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _sha256_file(path) != row["sha256"]
        ):
            raise RuntimeError(f"PrismaSnap collated metadata changed: {path}")
    census = _verify_output_census(
        root,
        source,
        compare_source_metadata=False,
    )
    expected = _build_collated_provenance(
        plan,
        census=census,
        receipts=receipts,
        changed_tensors=changed_tensors,
        part_bindings=part_bindings,
        metadata_bindings=metadata_bindings,
        transfer_strategy=transfer_strategy,
    )
    if provenance != expected:
        raise RuntimeError(
            "PrismaSnap collated checkpoint belongs to different ordered parts"
        )
    return provenance


def merge_checkpoint_parts(
    source_dir: str | Path,
    plan_dir: str | Path,
    part_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    resume: bool = False,
    require_hardlinks: bool = False,
) -> dict[str, object]:
    """Exact-union independently materialized shard parts into one HF model."""
    if len(part_dirs) < 2:
        raise ValueError("PrismaSnap checkpoint merge requires at least two parts")
    source = _Checkpoint(Path(source_dir), require_all_shards=False)
    plan, scales = load_plan(plan_dir)
    _validate_materialization_plan(plan, source, scales)

    def load_ordered_parts() -> list[tuple[Path, dict[str, object]]]:
        result: list[tuple[Path, dict[str, object]]] = []
        for raw_path in part_dirs:
            requested = Path(raw_path)
            if requested.is_symlink() or not requested.is_dir():
                raise RuntimeError(
                    f"PrismaSnap checkpoint part is not a real directory: {requested}"
                )
            root = requested.resolve(strict=True)
            result.append((root, _load_part(root, plan=plan, source=source)))
        return result

    parts = load_ordered_parts()
    part_bindings = [
        _part_collation_binding(ordinal, root, payload)
        for ordinal, (root, payload) in enumerate(parts)
    ]
    metadata_bindings = _collation_metadata_bindings(source)
    owners: dict[
        str, tuple[Path, dict[str, object], dict[str, object]]
    ] = {}
    receipts: list[dict[str, object]] = []
    changed = 0
    for ordinal, (root, payload) in enumerate(parts):
        binding = part_bindings[ordinal]
        changed += int(payload["changed_tensors"])
        for raw in payload["shards"]:
            row = dict(raw)
            name = str(row["source_name"])
            if name in owners:
                raise RuntimeError(f"PrismaSnap checkpoint parts overlap on {name}")
            owners[name] = (root, row, binding)
            receipts.append(row)
    if set(owners) != set(source.shards):
        raise RuntimeError(
            "PrismaSnap checkpoint parts do not exactly cover source shards; "
            f"missing={sorted(set(source.shards) - set(owners))[:12]}"
        )
    receipts.sort(key=lambda row: str(row["source_name"]))
    transfer_strategy = "hardlink_required" if require_hardlinks else "durable_copy"
    merge_state = _sealed_state(
        {
            "schema": PART_MERGE_STATE_SCHEMA,
            "state": "COPYING",
            "plan_sha256": str(plan["plan_sha256"]),
            "source_portable_content_sha256": _require_sha256(
                plan["source"]["portable_identity"]["portable_content_sha256"],
                where="PrismaSnap collation portable source identity",
            ),
            "ordered_part_bindings": part_bindings,
            "source_metadata": metadata_bindings,
            "shard_transfer_strategy": transfer_strategy,
        }
    )
    output = Path(output_dir)
    temporary = output.with_name(output.name + ".prismasnap-incomplete")

    def validate_tree(root: Path) -> dict[str, object]:
        result = _validate_collated_checkpoint_tree(
            root,
            source=source,
            plan=plan,
            receipts=receipts,
            changed_tensors=changed,
            part_bindings=part_bindings,
            metadata_bindings=metadata_bindings,
            transfer_strategy=transfer_strategy,
            allow_recovery_files=root == temporary,
        )
        if require_hardlinks:
            for name, (owner_root, _receipt, _binding) in owners.items():
                source_path = owner_root / name
                target = root / name
                source_stat = source_path.stat()
                target_stat = target.stat()
                if (
                    source_stat.st_dev != target_stat.st_dev
                    or source_stat.st_ino != target_stat.st_ino
                ):
                    raise RuntimeError(
                        f"PrismaSnap hardlink collation lost inode binding: {name}"
                    )
        return result

    def commit_ready() -> dict[str, object]:
        provenance = validate_tree(temporary)
        state_path = temporary / PART_MERGE_STATE_JSON
        if os.path.lexists(state_path):
            state_path.unlink()
        receipts_dir = temporary / PART_MERGE_RECEIPTS_DIR
        if os.path.lexists(receipts_dir):
            if receipts_dir.is_symlink() or not receipts_dir.is_dir():
                raise RuntimeError("unsafe PrismaSnap collation receipt directory")
            shutil.rmtree(receipts_dir)
        _fsync_dir(temporary)
        os.replace(temporary, output)
        _fsync_dir(output.parent)
        return provenance

    if os.path.lexists(output):
        if not resume or output.is_symlink() or not output.is_dir():
            raise RuntimeError(f"PrismaSnap merged checkpoint output exists: {output}")
        if os.path.lexists(temporary):
            raise RuntimeError(
                "PrismaSnap merged checkpoint has committed and staging trees"
            )
        return validate_tree(output)

    state_path = temporary / PART_MERGE_STATE_JSON
    if os.path.lexists(temporary):
        if not resume or temporary.is_symlink() or not temporary.is_dir():
            raise RuntimeError(
                f"stale PrismaSnap merged-checkpoint state exists: {temporary}"
            )
        _discard_interrupted_atomic_write(state_path, resume=True)
        _discard_interrupted_atomic_write(
            temporary / PROVENANCE_JSON, resume=True
        )
        if os.path.lexists(state_path):
            state = _load_json(state_path, where="PrismaSnap part merge state")
            _validate_sealed_state(
                state,
                merge_state,
                where="PrismaSnap part merge state",
            )
            if os.path.lexists(temporary / PROVENANCE_JSON):
                return commit_ready()
        elif os.path.lexists(temporary / PROVENANCE_JSON):
            return commit_ready()
        else:
            if any(temporary.iterdir()):
                raise RuntimeError(
                    "PrismaSnap checkpoint collation staging lacks durable state"
                )
            _atomic_json(state_path, merge_state)
            _fsync_dir(temporary)
    else:
        # Explicit resume is valid before a first attempt creates any state.
        temporary.mkdir(parents=True, exist_ok=False)
        _atomic_json(state_path, merge_state)
        _fsync_dir(temporary)

    receipts_dir = temporary / PART_MERGE_RECEIPTS_DIR
    if os.path.lexists(receipts_dir):
        if receipts_dir.is_symlink() or not receipts_dir.is_dir():
            raise RuntimeError("unsafe PrismaSnap collation receipt directory")
    else:
        receipts_dir.mkdir()
        _fsync_dir(temporary)
    expected_receipt_names = {f"{name}.json" for name in source.shards}
    for entry in receipts_dir.iterdir():
        expected_temporary_names = {
            f".{name}.json.tmp" for name in source.shards
        }
        if (
            entry.name not in expected_receipt_names | expected_temporary_names
            or entry.is_symlink()
            or not entry.is_file()
        ):
            raise RuntimeError(f"unsafe PrismaSnap collation receipt entry: {entry}")

    for name in source.shards:
        owner_root, expected_part_receipt, part_binding = owners[name]
        source_path = owner_root / name
        target = temporary / name
        receipt_path = receipts_dir / f"{name}.json"
        _discard_interrupted_atomic_write(receipt_path, resume=resume)
        expected_copy_receipt = {
            "schema": COLLATED_SHARD_RECEIPT_SCHEMA,
            "merge_state_sha256": merge_state["state_sha256"],
            "part_sha256": part_binding["part_sha256"],
            "part_manifest_file_sha256": part_binding["manifest_file_sha256"],
            "source_name": name,
            "source_file_sha256": expected_part_receipt["output_sha256"],
            "output_bytes": expected_part_receipt["output_bytes"],
            "output_sha256": expected_part_receipt["output_sha256"],
            "shard_transfer_strategy": transfer_strategy,
        }
        target_exists = os.path.lexists(target)
        receipt_exists = os.path.lexists(receipt_path)
        if target_exists and (target.is_symlink() or not target.is_file()):
            raise RuntimeError(f"unsafe PrismaSnap collated shard: {target}")
        if receipt_exists and (
            receipt_path.is_symlink() or not receipt_path.is_file()
        ):
            raise RuntimeError(f"unsafe PrismaSnap collated shard receipt: {receipt_path}")
        if receipt_exists:
            observed = _load_json(
                receipt_path, where="PrismaSnap collated shard receipt"
            )
            if observed != expected_copy_receipt:
                raise RuntimeError(f"PrismaSnap collated shard receipt equivocated: {name}")
            if not target_exists:
                raise RuntimeError(
                    f"PrismaSnap verified collated shard disappeared: {name}"
                )
        if target_exists:
            if (
                target.stat().st_size != int(expected_part_receipt["output_bytes"])
                or _sha256_file(target) != expected_part_receipt["output_sha256"]
            ):
                raise RuntimeError(f"PrismaSnap staged collated shard changed: {name}")
            if require_hardlinks:
                source_stat = source_path.stat()
                target_stat = target.stat()
                if (
                    source_stat.st_dev != target_stat.st_dev
                    or source_stat.st_ino != target_stat.st_ino
                ):
                    raise RuntimeError(
                        f"PrismaSnap staged shard is not the required hardlink: {name}"
                    )
            if not receipt_exists:
                # Recover the atomic shard-rename / receipt-write window only
                # after revalidating the exact expected bytes.
                _atomic_json(receipt_path, expected_copy_receipt)
                _fsync_dir(receipts_dir)
                _fsync_dir(temporary)
            print(f"[prismasnap-collate] resume verified {name}", flush=True)
            continue

        before = source_path.stat()
        if require_hardlinks:
            staging_device = temporary.stat().st_dev
            if before.st_dev != staging_device:
                raise RuntimeError(
                    f"PrismaSnap required hardlink crosses filesystems: {source_path}"
                )
            try:
                os.link(source_path, target, follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(
                    f"PrismaSnap required hardlink failed for {source_path}"
                ) from exc
            _fsync_dir(temporary)
        else:
            _copy_file_durable(source_path, target)
        after = source_path.stat()
        target_stat = target.stat()
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or target_stat.st_size != int(expected_part_receipt["output_bytes"])
            or _sha256_file(target) != expected_part_receipt["output_sha256"]
            or (
                require_hardlinks
                and (
                    target_stat.st_dev != after.st_dev
                    or target_stat.st_ino != after.st_ino
                )
            )
        ):
            raise RuntimeError(
                f"PrismaSnap merged shard changed during transfer: {source_path}"
            )
        _atomic_json(receipt_path, expected_copy_receipt)
        _fsync_dir(receipts_dir)
        _fsync_dir(temporary)

    # Re-open every part after the copies.  A mutable worker directory may not
    # change between admission and final provenance publication.
    reloaded = load_ordered_parts()
    reloaded_bindings = [
        _part_collation_binding(ordinal, root, payload)
        for ordinal, (root, payload) in enumerate(reloaded)
    ]
    if reloaded_bindings != part_bindings:
        raise RuntimeError("PrismaSnap checkpoint parts changed during collation")
    if _collation_metadata_bindings(source) != metadata_bindings:
        raise RuntimeError("PrismaSnap source metadata changed during collation")
    _copy_checkpoint_metadata(source, temporary)
    _validate_config_semantics(temporary, plan["source"]["identity"])
    census = _verify_output_census(
        temporary, source, compare_source_metadata=False
    )
    provenance = _build_collated_provenance(
        plan,
        census=census,
        receipts=receipts,
        changed_tensors=changed,
        part_bindings=part_bindings,
        metadata_bindings=metadata_bindings,
        transfer_strategy=transfer_strategy,
    )
    _atomic_json(temporary / PROVENANCE_JSON, provenance)
    expected_staging_entries = {
        PART_MERGE_STATE_JSON,
        PART_MERGE_RECEIPTS_DIR,
        PROVENANCE_JSON,
        *source.shards,
        *(str(row["name"]) for row in metadata_bindings),
    }
    actual_staging_entries = {item.name for item in temporary.iterdir()}
    if actual_staging_entries != expected_staging_entries:
        raise RuntimeError(
            "PrismaSnap collation staging file census differs: "
            f"extra={sorted(actual_staging_entries - expected_staging_entries)}"
        )
    if validate_tree(temporary) != provenance:
        raise RuntimeError("PrismaSnap collated provenance changed after publication")
    state_path.unlink()
    shutil.rmtree(receipts_dir)
    _fsync_dir(temporary)
    os.replace(temporary, output)
    _fsync_dir(output.parent)
    return provenance
