"""Strict, pread-only BF16 corpus contract for trellis research.

The first GLM-5.3 corpus captured valid BF16 weight bytes but published an
explicitly incomplete manifest and no activation-derived importance vectors.
This module deliberately does not make that artifact loadable.  It provides:

* a fail-closed finalized ``trellis.bf16_corpus.v2`` loader;
* the one canonical adapter from the existing sensitivity-probe marginals;
* an immutable finalizer that copies the old weight payload byte-for-byte into
  a new safetensors artifact and appends FP32 importance vectors.

All safetensors reads use ``os.pread``.  This is an offline corpus-build path,
not a production cache or a replacement for any resident prefetch mechanism.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .cb_imatrix import canonical_imatrix_sha256


FINALIZED_SCHEMA = "trellis.bf16_corpus.v2"
INCOMPLETE_GLM_SCHEMA = "trellis.glm_corpus.v0-INCOMPLETE"
IMPORTANCE_SCHEMA = "prismaquant.glm_trellis_importance.probe_imatrix.v1"
READER_SCHEMA = "prismaquant.trellis_bf16_corpus.pread.v1"

GLM_MODEL_PROFILE = "glm5_next"
GLM_NUM_HIDDEN_LAYERS = 45
GLM_DENSE_LAYERS = (0, 1, 2)
GLM_ROUTED_LAYERS = (3, 9, 15, 21, 26, 32, 38, 44)
GLM_LAYERS = tuple(sorted((*GLM_DENSE_LAYERS, *GLM_ROUTED_LAYERS)))
GLM_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
GLM_DENSE_COUNT = len(GLM_DENSE_LAYERS) * len(GLM_PROJECTIONS)
GLM_ROUTED_COUNT = len(GLM_ROUTED_LAYERS) * len(GLM_PROJECTIONS)
GLM_ENTRY_COUNT = GLM_DENSE_COUNT + GLM_ROUTED_COUNT
GLM_EXPERT = 0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DENSE_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>[0-9]+)\.mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_ROUTED_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_DTYPE_BYTES = {"BF16": 2, "F32": 4}


class CorpusContractError(ValueError):
    """A corpus or probe does not satisfy the finalized contract."""


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CorpusContractError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw, object_pairs_hook=_no_duplicate_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusContractError(f"{path}: expected a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY)
    try:
        offset = 0
        size = os.fstat(descriptor).st_size
        while offset < size:
            block = os.pread(descriptor, min(8 << 20, size - offset), offset)
            if not block:
                raise CorpusContractError(f"{path}: unexpected EOF at {offset}")
            digest.update(block)
            offset += len(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _require_sha256(value: object, *, where: str) -> str:
    text = str(value)
    if not _SHA256_RE.fullmatch(text):
        raise CorpusContractError(f"{where} must be a lowercase SHA-256")
    return text


def _require_calibration_hash(value: object, *, where: str) -> str:
    """Validate PrismaQuant's existing calibration-data identity.

    ``calibration_data_hash`` is BLAKE2b-128, not SHA-256.  Keeping this
    distinct from file and descriptor SHA-256 fields prevents a finalized
    corpus from relabelling a real probe's 32-hex identity as another hash.
    """
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{32}", text) is None:
        raise CorpusContractError(f"{where} must be lowercase BLAKE2b-128 hex")
    return text


def _require_positive_int(value: object, *, where: str) -> int:
    if isinstance(value, bool):
        raise CorpusContractError(f"{where} must be a positive integer")
    try:
        parsed = int(value)
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise CorpusContractError(f"{where} must be a positive integer") from exc
    if not exact or parsed <= 0:
        raise CorpusContractError(f"{where} must be a positive integer")
    return parsed


def _require_nonnegative_int(value: object, *, where: str) -> int:
    if isinstance(value, bool):
        raise CorpusContractError(f"{where} must be a nonnegative integer")
    try:
        parsed = int(value)
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise CorpusContractError(f"{where} must be a nonnegative integer") from exc
    if not exact or parsed < 0:
        raise CorpusContractError(f"{where} must be a nonnegative integer")
    return parsed


@dataclass(frozen=True)
class _SafetensorsLayout:
    path: Path
    data_start: int
    data_bytes: int
    tensors: Mapping[str, Mapping[str, Any]]


def _read_safetensors_layout(path: Path) -> _SafetensorsLayout:
    """Read and validate a safetensors header without mmap."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise CorpusContractError(f"cannot open corpus {path}: {exc}") from exc
    try:
        size = os.fstat(descriptor).st_size
        raw_length = os.pread(descriptor, 8, 0)
        if len(raw_length) != 8:
            raise CorpusContractError(f"{path}: truncated safetensors prefix")
        header_length = int.from_bytes(raw_length, "little", signed=False)
        if header_length <= 0 or header_length > size - 8:
            raise CorpusContractError(
                f"{path}: invalid safetensors header length {header_length}"
            )
        raw_header = os.pread(descriptor, header_length, 8)
        if len(raw_header) != header_length:
            raise CorpusContractError(f"{path}: truncated safetensors header")
        try:
            header = json.loads(
                raw_header.decode("utf-8"), object_pairs_hook=_no_duplicate_object
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorpusContractError(f"{path}: invalid safetensors JSON: {exc}") from exc
        if not isinstance(header, dict):
            raise CorpusContractError(f"{path}: safetensors header is not an object")
        header.pop("__metadata__", None)
        data_bytes = size - 8 - header_length
        intervals: list[tuple[int, int, str]] = []
        for name, raw in header.items():
            if not isinstance(name, str) or not name or not isinstance(raw, dict):
                raise CorpusContractError(f"{path}: invalid tensor header {name!r}")
            dtype = raw.get("dtype")
            shape = raw.get("shape")
            offsets = raw.get("data_offsets")
            if dtype not in _DTYPE_BYTES:
                raise CorpusContractError(f"{path}: {name}: unsupported dtype {dtype!r}")
            if (
                not isinstance(shape, list)
                or not shape
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)
            ):
                raise CorpusContractError(f"{path}: {name}: invalid shape {shape!r}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in offsets)
            ):
                raise CorpusContractError(f"{path}: {name}: invalid data_offsets")
            start, end = offsets
            expected = math.prod(shape) * _DTYPE_BYTES[dtype]
            if start < 0 or end <= start or end - start != expected or end > data_bytes:
                raise CorpusContractError(
                    f"{path}: {name}: invalid byte span {offsets!r}, expected {expected} bytes"
                )
            intervals.append((start, end, name))
        if not intervals:
            raise CorpusContractError(f"{path}: no tensors")
        intervals.sort()
        cursor = 0
        for start, end, name in intervals:
            if start != cursor:
                raise CorpusContractError(
                    f"{path}: tensor payload is not contiguous before {name!r}"
                )
            cursor = end
        if cursor != data_bytes:
            raise CorpusContractError(
                f"{path}: unclaimed payload bytes: claimed={cursor}, file={data_bytes}"
            )
        return _SafetensorsLayout(
            path=path,
            data_start=8 + header_length,
            data_bytes=data_bytes,
            tensors=header,
        )
    finally:
        os.close(descriptor)


def _tensor_bytes(layout: _SafetensorsLayout, name: str) -> bytes:
    try:
        start, end = layout.tensors[name]["data_offsets"]
    except KeyError as exc:
        raise CorpusContractError(f"{layout.path}: missing tensor {name!r}") from exc
    descriptor = os.open(layout.path, os.O_RDONLY)
    try:
        value = os.pread(descriptor, end - start, layout.data_start + start)
    finally:
        os.close(descriptor)
    if len(value) != end - start:
        raise CorpusContractError(f"{layout.path}: truncated tensor {name!r}")
    return value


def _tensor_from_bytes(raw: bytes, *, dtype: str, shape: Sequence[int]) -> torch.Tensor:
    # bytearray supplies a writable owner, avoiding torch.frombuffer's warning
    # and ensuring the returned tensor does not alias a transient bytes object.
    storage = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    if dtype == "BF16":
        value = storage.view(torch.bfloat16)
    elif dtype == "F32":
        value = storage.view(torch.float32)
    else:  # layout validation makes this unreachable
        raise CorpusContractError(f"unsupported tensor dtype {dtype!r}")
    return value.reshape(tuple(int(v) for v in shape)).clone()


def _expected_shape(population: str, projection: str) -> tuple[int, int]:
    if population == "dense":
        return (4096, 12288) if projection == "down_proj" else (12288, 4096)
    if population == "routed":
        return (4096, 2048) if projection == "down_proj" else (2048, 4096)
    raise CorpusContractError(f"unknown population {population!r}")


def _classify_glm_name(name: str) -> tuple[str, int, str, int | None]:
    routed = _ROUTED_RE.fullmatch(name)
    if routed:
        return (
            "routed",
            int(routed.group("layer")),
            routed.group("projection"),
            int(routed.group("expert")),
        )
    dense = _DENSE_RE.fullmatch(name)
    if dense:
        return "dense", int(dense.group("layer")), dense.group("projection"), None
    raise CorpusContractError(f"not a GLM MLP corpus tensor name: {name!r}")


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    population: str
    layer: int
    projection: str
    expert: int | None
    source_weight_shape: tuple[int, int]
    source_weight_sha256: str
    importance_key: str
    importance_shape: tuple[int]
    importance_sha256: str
    importance_source_qname: str
    importance_source_expert: int | None
    importance_denominator_name: str
    importance_denominator: int


@dataclass(frozen=True)
class FinalizedBF16Corpus:
    manifest_path: Path
    artifact_path: Path
    manifest: Mapping[str, Any]
    entries: tuple[CorpusEntry, ...]
    _layout: _SafetensorsLayout

    def load_tensor(self, entry: CorpusEntry | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one BF16 weight and its FP32 importance via pread."""

        if isinstance(entry, str):
            matches = [candidate for candidate in self.entries if candidate.name == entry]
            if len(matches) != 1:
                raise CorpusContractError(f"unknown corpus entry {entry!r}")
            entry = matches[0]
        weight_raw = _tensor_bytes(self._layout, entry.name)
        importance_raw = _tensor_bytes(self._layout, entry.importance_key)
        return (
            _tensor_from_bytes(weight_raw, dtype="BF16", shape=entry.source_weight_shape),
            _tensor_from_bytes(importance_raw, dtype="F32", shape=entry.importance_shape),
        )

    @property
    def populations(self) -> Mapping[str, tuple[CorpusEntry, ...]]:
        return {
            name: tuple(entry for entry in self.entries if entry.population == name)
            for name in ("dense", "routed")
        }


def _manifest_file(root: Path, value: object) -> Path:
    name = str(value)
    if not name or Path(name).name != name:
        raise CorpusContractError("manifest file must be a same-directory basename")
    path = root / name
    if not path.is_file():
        raise CorpusContractError(f"corpus file does not exist: {path}")
    return path


def load_finalized_bf16_corpus(
    manifest_path: str | Path,
    *,
    verify_file_hash: bool = True,
) -> FinalizedBF16Corpus:
    """Validate and open the finalized GLM BF16 corpus.

    This validation intentionally includes the exact experimental census.  A
    routed/dense mix with a different layer selection is a different corpus,
    not an optional extension of this one.
    """

    path = Path(manifest_path).resolve()
    manifest = _read_json_object(path)
    schema = manifest.get("schema")
    if schema != FINALIZED_SCHEMA:
        if "INCOMPLETE" in str(schema).upper():
            raise CorpusContractError(f"{path}: incomplete corpus is never loadable")
        raise CorpusContractError(f"{path}: expected schema {FINALIZED_SCHEMA!r}")
    if manifest.get("status") != "finalized":
        raise CorpusContractError(f"{path}: status must be 'finalized'")
    if manifest.get("model_profile") != GLM_MODEL_PROFILE:
        raise CorpusContractError(f"{path}: model_profile must be {GLM_MODEL_PROFILE!r}")
    if manifest.get("num_hidden_layers") != GLM_NUM_HIDDEN_LAYERS:
        raise CorpusContractError(f"{path}: num_hidden_layers must be 45")
    if tuple(manifest.get("layers", ())) != GLM_LAYERS:
        raise CorpusContractError(f"{path}: selected layer census differs")
    if tuple(manifest.get("roles", ())) != GLM_PROJECTIONS:
        raise CorpusContractError(f"{path}: projection census differs")
    if manifest.get("expert") != GLM_EXPERT:
        raise CorpusContractError(f"{path}: selected expert must be 0")
    if not isinstance(manifest.get("model"), str) or not manifest["model"]:
        raise CorpusContractError(f"{path}: model must be a nonempty source identity")
    _require_sha256(manifest.get("model_config_sha256"), where="model_config_sha256")
    commit = str(manifest.get("prismaquant_commit"))
    if not _COMMIT_RE.fullmatch(commit):
        raise CorpusContractError("prismaquant_commit must be a lowercase 40-hex commit")

    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict) or not calibration:
        raise CorpusContractError("calibration must be a nonempty object")
    calibration = _canonical_calibration(calibration)
    calibration_hash = _require_calibration_hash(
        calibration.get("probe_calib_hash"),
        where="calibration.probe_calib_hash",
    )
    if not isinstance(calibration.get("dataset"), str) or not calibration["dataset"]:
        raise CorpusContractError("calibration.dataset must be a nonempty string")
    for field in ("nsamples", "seqlen", "tokens"):
        _require_positive_int(calibration.get(field), where=f"calibration.{field}")
    _require_nonnegative_int(calibration.get("seed"), where="calibration.seed")
    identity = manifest.get("importance_identity")
    if not isinstance(identity, dict) or identity.get("schema") != IMPORTANCE_SCHEMA:
        raise CorpusContractError("importance_identity schema is missing or invalid")
    if identity.get("probe_calibration_hash") != calibration_hash:
        raise CorpusContractError("probe/calibration identities differ")
    _require_sha256(identity.get("probe_file_sha256"), where="probe_file_sha256")
    _require_sha256(
        identity.get("probe_imatrix_value_sha256"),
        where="probe_imatrix_value_sha256",
    )
    declared_value_hash = _require_sha256(
        identity.get("value_sha256"), where="importance value_sha256"
    )
    expected_identity_text = {
        "dense_normalization": "act_sq_sum / n_tokens_seen",
        "routed_normalization": "expert_act_sq_sum[expert] / expert_tokens[expert]",
        "gate_up_mapping": "expert 0 gate_up_proj -> expert 0 gate_proj and up_proj",
        "down_mapping": "expert 0 down_proj -> expert 0 down_proj",
    }
    for field, expected in expected_identity_text.items():
        if identity.get(field) != expected:
            raise CorpusContractError(f"importance_identity.{field} differs")
    reader = manifest.get("reader_contract")
    if reader != {
        "schema": READER_SCHEMA,
        "method": "os.pread",
        "mmap": False,
        "source_weight_dtype": "torch.bfloat16",
        "importance_dtype": "torch.float32",
        "consumer_weight_dtype": "torch.float32",
    }:
        raise CorpusContractError("reader_contract must be the exact pread-only contract")

    populations = manifest.get("populations")
    expected_populations = {
        "dense": {"count": GLM_DENSE_COUNT, "layers": list(GLM_DENSE_LAYERS)},
        "routed": {"count": GLM_ROUTED_COUNT, "layers": list(GLM_ROUTED_LAYERS)},
    }
    if populations != expected_populations:
        raise CorpusContractError("population census differs from the exact GLM contract")
    source_artifact = manifest.get("source_artifact")
    if not isinstance(source_artifact, dict):
        raise CorpusContractError("source_artifact must be an object")
    if source_artifact.get("manifest_schema") != INCOMPLETE_GLM_SCHEMA:
        raise CorpusContractError("source_artifact manifest schema differs")
    _require_sha256(
        source_artifact.get("file_sha256"), where="source_artifact.file_sha256"
    )
    _require_positive_int(
        source_artifact.get("payload_bytes"), where="source_artifact.payload_bytes"
    )
    if source_artifact.get("payload_copy") != "byte-for-byte via os.pread":
        raise CorpusContractError("source_artifact payload-copy contract differs")

    artifact_path = _manifest_file(path.parent, manifest.get("file"))
    expected_size = _require_positive_int(manifest.get("file_size_bytes"), where="file_size_bytes")
    if artifact_path.stat().st_size != expected_size:
        raise CorpusContractError("corpus file size differs from manifest")
    expected_file_hash = _require_sha256(manifest.get("file_sha256"), where="file_sha256")
    if verify_file_hash and _sha256_file(artifact_path) != expected_file_hash:
        raise CorpusContractError("corpus file SHA-256 differs from manifest")
    layout = _read_safetensors_layout(artifact_path)

    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != GLM_ENTRY_COUNT:
        raise CorpusContractError(f"entries must contain exactly {GLM_ENTRY_COUNT} rows")
    entries: list[CorpusEntry] = []
    seen_names: set[str] = set()
    seen_importance: set[str] = set()
    population_counts = {"dense": 0, "routed": 0}
    importance_values: dict[str, torch.Tensor] = {}
    for index, raw in enumerate(raw_entries):
        where = f"entries[{index}]"
        if not isinstance(raw, dict):
            raise CorpusContractError(f"{where} must be an object")
        name = str(raw.get("name"))
        importance_key = str(raw.get("importance_key"))
        if name in seen_names:
            raise CorpusContractError(f"duplicate corpus tensor {name!r}")
        if importance_key in seen_importance:
            raise CorpusContractError(f"duplicate importance key {importance_key!r}")
        seen_names.add(name)
        seen_importance.add(importance_key)
        population, layer, projection, expert = _classify_glm_name(name)
        if raw.get("population") != population:
            raise CorpusContractError(f"{where}: population label differs from name")
        if raw.get("layer") != layer or raw.get("projection") != projection:
            raise CorpusContractError(f"{where}: layer/projection differs from name")
        if raw.get("expert") != expert:
            raise CorpusContractError(f"{where}: expert differs from name")
        allowed_layers = GLM_DENSE_LAYERS if population == "dense" else GLM_ROUTED_LAYERS
        if layer not in allowed_layers or (population == "routed" and expert != GLM_EXPERT):
            raise CorpusContractError(f"{where}: tensor is outside the selected census")
        expected_shape = _expected_shape(population, projection)
        if tuple(raw.get("source_weight_shape", ())) != expected_shape:
            raise CorpusContractError(f"{where}: source weight shape differs")
        if raw.get("source_weight_dtype") != "torch.bfloat16":
            raise CorpusContractError(f"{where}: source weight must be torch.bfloat16")
        source_hash = _require_sha256(
            raw.get("source_weight_sha256"),
            where=f"{where}.source_weight_sha256",
        )
        if importance_key != f"__bf16_importance__.{name}":
            raise CorpusContractError(f"{where}: noncanonical importance key")
        importance_shape = (expected_shape[1],)
        if tuple(raw.get("importance_shape", ())) != importance_shape:
            raise CorpusContractError(f"{where}: importance shape differs")
        if raw.get("importance_dtype") != "torch.float32":
            raise CorpusContractError(f"{where}: importance must be torch.float32")
        importance_hash = _require_sha256(
            raw.get("importance_sha256"), where=f"{where}.importance_sha256"
        )
        source = raw.get("importance_source")
        if not isinstance(source, dict):
            raise CorpusContractError(f"{where}: importance_source must be an object")
        source_qname = str(source.get("qname"))
        source_expert = source.get("expert")
        expected_source_expert = GLM_EXPERT if population == "routed" else None
        if source_expert != expected_source_expert:
            raise CorpusContractError(f"{where}: importance source expert differs")
        if population == "dense":
            expected_source_qname = name.removesuffix(".weight")
        else:
            prefix = name.split(".experts.0.", 1)[0]
            packed_projection = (
                "down_proj" if projection == "down_proj" else "gate_up_proj"
            )
            expected_source_qname = f"{prefix}.experts.{packed_projection}"
        if source_qname != expected_source_qname:
            raise CorpusContractError(f"{where}: importance source qname differs")
        denominator_name = str(source.get("denominator_name"))
        expected_denominator_name = "expert_tokens" if population == "routed" else "n_tokens_seen"
        if denominator_name != expected_denominator_name:
            raise CorpusContractError(f"{where}: importance denominator kind differs")
        denominator = _require_positive_int(
            source.get("denominator"),
            where=f"{where}.importance_source.denominator",
        )
        weight_header = layout.tensors.get(name)
        importance_header = layout.tensors.get(importance_key)
        if weight_header is None or importance_header is None:
            raise CorpusContractError(f"{where}: tensor or importance missing from artifact")
        if (
            weight_header.get("dtype") != "BF16"
            or tuple(weight_header.get("shape", ())) != expected_shape
        ):
            raise CorpusContractError(f"{where}: BF16 artifact header differs")
        if (
            importance_header.get("dtype") != "F32"
            or tuple(importance_header.get("shape", ())) != importance_shape
        ):
            raise CorpusContractError(f"{where}: importance artifact header differs")
        weight_raw = _tensor_bytes(layout, name)
        importance_raw = _tensor_bytes(layout, importance_key)
        if hashlib.sha256(weight_raw).hexdigest() != source_hash:
            raise CorpusContractError(f"{where}: source weight hash differs")
        if hashlib.sha256(importance_raw).hexdigest() != importance_hash:
            raise CorpusContractError(f"{where}: importance hash differs")
        importance = _tensor_from_bytes(importance_raw, dtype="F32", shape=importance_shape)
        if not bool(torch.isfinite(importance).all()):
            raise CorpusContractError(f"{where}: importance is non-finite")
        if bool((importance < 0).any()):
            raise CorpusContractError(f"{where}: importance is negative")
        importance_values[name] = importance
        entries.append(CorpusEntry(
            name=name,
            population=population,
            layer=layer,
            projection=projection,
            expert=expert,
            source_weight_shape=expected_shape,
            source_weight_sha256=source_hash,
            importance_key=importance_key,
            importance_shape=importance_shape,
            importance_sha256=importance_hash,
            importance_source_qname=source_qname,
            importance_source_expert=source_expert,
            importance_denominator_name=denominator_name,
            importance_denominator=denominator,
        ))
        population_counts[population] += 1

    if population_counts != {"dense": GLM_DENSE_COUNT, "routed": GLM_ROUTED_COUNT}:
        raise CorpusContractError("dense/routed population counts differ")
    expected_names = {entry.name for entry in entries} | {
        entry.importance_key for entry in entries
    }
    if set(layout.tensors) != expected_names:
        raise CorpusContractError("artifact tensor census differs from manifest")
    if canonical_imatrix_sha256(importance_values) != declared_value_hash:
        raise CorpusContractError("importance value hash differs from manifest")
    return FinalizedBF16Corpus(path, artifact_path, manifest, tuple(entries), layout)


@dataclass(frozen=True)
class ImportanceVector:
    target_name: str
    value: torch.Tensor
    source_qname: str
    source_expert: int | None
    denominator_name: str
    denominator: int


def validate_incomplete_glm_source(
    incomplete_manifest_path: str | Path,
    source_artifact_path: str | Path,
) -> dict[str, object]:
    """Validate the sound subset of v0 without making it consumer-loadable."""

    incomplete_path = Path(incomplete_manifest_path).resolve()
    source_path = Path(source_artifact_path).resolve()
    incomplete = _read_json_object(incomplete_path)
    if incomplete.get("schema") != INCOMPLETE_GLM_SCHEMA:
        raise CorpusContractError("source manifest is not the explicit incomplete GLM artifact")
    raw_tensors = incomplete.get("tensors")
    if not isinstance(raw_tensors, dict) or len(raw_tensors) != GLM_ENTRY_COUNT:
        raise CorpusContractError("source manifest must contain exactly 33 tensors")
    declared_hash = _require_sha256(
        incomplete.get("file_sha256"), where="incomplete file_sha256"
    )
    actual_hash = _sha256_file(source_path)
    if actual_hash != declared_hash:
        raise CorpusContractError("source artifact hash differs from incomplete manifest")
    layout = _read_safetensors_layout(source_path)
    if set(layout.tensors) != set(raw_tensors):
        raise CorpusContractError("source artifact tensor census differs from incomplete manifest")
    counts = {"dense": 0, "routed": 0}
    for name, descriptor in layout.tensors.items():
        population, layer, projection, expert = _classify_glm_name(name)
        allowed_layers = GLM_DENSE_LAYERS if population == "dense" else GLM_ROUTED_LAYERS
        expected_shape = _expected_shape(population, projection)
        raw = raw_tensors[name]
        if (
            layer not in allowed_layers
            or (population == "routed" and expert != GLM_EXPERT)
            or descriptor.get("dtype") != "BF16"
            or tuple(descriptor.get("shape", ())) != expected_shape
            or not isinstance(raw, dict)
            or tuple(raw.get("shape", ())) != expected_shape
            or raw.get("role") != population
        ):
            raise CorpusContractError(f"{name}: incomplete source contract differs")
        counts[population] += 1
    if counts != {"dense": GLM_DENSE_COUNT, "routed": GLM_ROUTED_COUNT}:
        raise CorpusContractError("incomplete source population census differs")
    return {
        "manifest_schema": INCOMPLETE_GLM_SCHEMA,
        "file_sha256": actual_hash,
        "payload_bytes": layout.data_bytes,
        "population_counts": counts,
    }


def _probe_stats(
    payload: object, *, source: Path
) -> tuple[Mapping[str, Mapping[str, object]], Mapping[str, object]]:
    if not isinstance(payload, Mapping):
        raise CorpusContractError(f"{source}: probe must be a mapping")
    raw_stats = payload.get("stats", payload)
    if not isinstance(raw_stats, Mapping):
        raise CorpusContractError(f"{source}: probe stats must be a mapping")
    stats = {
        str(name): value
        for name, value in raw_stats.items()
        if name != "meta" and isinstance(value, Mapping)
    }
    meta = payload.get("meta", {})
    if not isinstance(meta, Mapping):
        raise CorpusContractError(f"{source}: probe meta must be a mapping")
    return stats, meta


def _calibration_hash(meta: Mapping[str, object], *, source: Path) -> str:
    values = {
        str(meta[key])
        for key in ("calib_hash", "calibration_hash", "calib_sha256")
        if meta.get(key)
    }
    if len(values) != 1:
        raise CorpusContractError(
            f"{source}: probe must carry exactly one coherent calibration hash"
        )
    return _require_calibration_hash(
        next(iter(values)), where="probe calibration hash"
    )


def adapt_glm_importance_from_probe(
    incomplete_manifest_path: str | Path,
    probe_path: str | Path,
) -> tuple[dict[str, ImportanceVector], dict[str, object]]:
    """Map the existing packed/dense probe imatrix onto the 33 GLM weights."""

    incomplete_path = Path(incomplete_manifest_path).resolve()
    incomplete = _read_json_object(incomplete_path)
    if incomplete.get("schema") != INCOMPLETE_GLM_SCHEMA:
        raise CorpusContractError(
            f"{incomplete_path}: expected the explicit incomplete GLM schema"
        )
    raw_tensors = incomplete.get("tensors")
    if not isinstance(raw_tensors, dict) or len(raw_tensors) != GLM_ENTRY_COUNT:
        raise CorpusContractError("incomplete GLM tensor census must contain 33 rows")

    probe = Path(probe_path).resolve()
    try:
        with probe.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.UnpicklingError) as exc:
        raise CorpusContractError(f"cannot read trusted probe {probe}: {exc}") from exc
    stats, meta = _probe_stats(payload, source=probe)
    calibration_hash = _calibration_hash(meta, source=probe)
    result: dict[str, ImportanceVector] = {}
    source_values: dict[str, torch.Tensor] = {}
    for name in sorted(raw_tensors):
        population, layer, projection, expert = _classify_glm_name(name)
        expected_shape = _expected_shape(population, projection)
        raw_meta = raw_tensors[name]
        if not isinstance(raw_meta, dict) or tuple(raw_meta.get("shape", ())) != expected_shape:
            raise CorpusContractError(f"{name}: incomplete-manifest shape differs")
        if raw_meta.get("role") != population:
            raise CorpusContractError(f"{name}: incomplete-manifest population differs")
        if population == "dense":
            source_qname = name.removesuffix(".weight")
            source_expert = None
            denominator_name = "n_tokens_seen"
            raw_stat = stats.get(source_qname)
            denominator_raw = raw_stat.get(denominator_name) if raw_stat else None
            numerator = raw_stat.get("act_sq_sum") if raw_stat else None
            value = (
                torch.as_tensor(numerator, dtype=torch.float32).detach().cpu()
                if numerator is not None else None
            )
        else:
            prefix = name.split(".experts.0.", 1)[0]
            packed_projection = "down_proj" if projection == "down_proj" else "gate_up_proj"
            source_qname = f"{prefix}.experts.{packed_projection}"
            source_expert = expert
            denominator_name = "expert_tokens"
            raw_stat = stats.get(source_qname)
            sums_raw = raw_stat.get("expert_act_sq_sum") if raw_stat else None
            tokens = raw_stat.get(denominator_name) if raw_stat else None
            sums = (
                torch.as_tensor(sums_raw, dtype=torch.float32).detach().cpu()
                if sums_raw is not None else None
            )
            token_values = (
                torch.as_tensor(tokens).detach().cpu()
                if tokens is not None else None
            )
            if (
                sums is None or token_values is None
                or sums.ndim != 2 or token_values.ndim != 1
                or sums.shape[0] != token_values.shape[0]
                or source_expert is None or source_expert >= sums.shape[0]
            ):
                raise CorpusContractError(
                    f"{name}: packed probe source has invalid expert shapes"
                )
            if (
                token_values.dtype == torch.bool
                or not bool(torch.isfinite(token_values).all())
                or not torch.equal(
                    token_values, token_values.to(torch.int64).to(token_values.dtype)
                )
                or bool((token_values < 0).any())
            ):
                raise CorpusContractError(
                    f"{name}: expert_tokens must be finite nonnegative integers"
                )
            if not bool(torch.isfinite(sums).all()) or bool((sums < 0).any()):
                raise CorpusContractError(
                    f"{name}: expert_act_sq_sum must be finite and nonnegative"
                )
            denominator_raw = (
                token_values[source_expert].item()
            )
            value = sums[source_expert]
        if raw_stat is None or value is None:
            raise CorpusContractError(f"{name}: missing probe source {source_qname!r}")
        denominator = _require_positive_int(
            denominator_raw, where=f"{name} importance denominator"
        )
        vector = (
            torch.as_tensor(value, dtype=torch.float32)
            .detach().cpu().reshape(-1).contiguous()
        ) / float(denominator)
        if vector.numel() != expected_shape[1]:
            raise CorpusContractError(
                f"{name}: importance width {vector.numel()} differs from {expected_shape[1]}"
            )
        if not bool(torch.isfinite(vector).all()) or bool((vector < 0).any()):
            raise CorpusContractError(f"{name}: importance must be finite and nonnegative")
        result[name] = ImportanceVector(
            target_name=name,
            value=vector,
            source_qname=source_qname,
            source_expert=source_expert,
            denominator_name=denominator_name,
            denominator=denominator,
        )
        source_key = (
            source_qname if source_expert is None
            else f"{source_qname}#expert={source_expert}"
        )
        previous = source_values.get(source_key)
        if previous is not None and not torch.equal(previous, vector):
            raise CorpusContractError(
                f"{name}: repeated probe source normalized inconsistently"
            )
        source_values[source_key] = vector
    counts = {
        population: sum(_classify_glm_name(name)[0] == population for name in result)
        for population in ("dense", "routed")
    }
    if counts != {"dense": GLM_DENSE_COUNT, "routed": GLM_ROUTED_COUNT}:
        raise CorpusContractError("adapted importance population census differs")
    value_hash = canonical_imatrix_sha256(
        {name: vector.value for name, vector in result.items()}
    )
    return result, {
        "schema": IMPORTANCE_SCHEMA,
        "probe_file_sha256": _sha256_file(probe),
        "probe_calibration_hash": calibration_hash,
        "probe_imatrix_value_sha256": canonical_imatrix_sha256(source_values),
        "value_sha256": value_hash,
        "dense_normalization": "act_sq_sum / n_tokens_seen",
        "routed_normalization": "expert_act_sq_sum[expert] / expert_tokens[expert]",
        "gate_up_mapping": "expert 0 gate_up_proj -> expert 0 gate_proj and up_proj",
        "down_mapping": "expert 0 down_proj -> expert 0 down_proj",
    }


def _encode_f32(value: torch.Tensor) -> bytes:
    normalized = value.detach().cpu().to(torch.float32).contiguous()
    return normalized.numpy().astype("<f4", copy=False).tobytes(order="C")


def _safetensors_header_bytes(header: Mapping[str, object]) -> bytes:
    raw = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    padding = (-len(raw)) % 8
    return raw + (b" " * padding)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _copy_payload_pread(source: _SafetensorsLayout, destination_fd: int) -> None:
    source_fd = os.open(source.path, os.O_RDONLY)
    try:
        offset = 0
        while offset < source.data_bytes:
            block = os.pread(
                source_fd,
                min(8 << 20, source.data_bytes - offset),
                source.data_start + offset,
            )
            if not block:
                raise CorpusContractError(
                    f"{source.path}: unexpected EOF while copying payload at {offset}"
                )
            _write_all(destination_fd, block)
            offset += len(block)
    finally:
        os.close(source_fd)


def _canonical_calibration(calibration: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(calibration, Mapping) or not calibration:
        raise CorpusContractError("calibration must be a nonempty mapping")
    out = dict(calibration)
    identity_sha256 = _require_sha256(
        out.get("identity_sha256"), where="calibration.identity_sha256"
    )
    out["probe_calib_hash"] = _require_calibration_hash(
        out.get("probe_calib_hash"), where="calibration.probe_calib_hash"
    )
    if not isinstance(out.get("dataset"), str) or not out["dataset"]:
        raise CorpusContractError("calibration.dataset must be a nonempty string")
    for field in ("nsamples", "seqlen", "tokens"):
        out[field] = _require_positive_int(out.get(field), where=f"calibration.{field}")
    out["seed"] = _require_nonnegative_int(out.get("seed"), where="calibration.seed")
    identity_payload = {key: value for key, value in out.items()
                        if key != "identity_sha256"}
    expected_identity = hashlib.sha256(json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")).hexdigest()
    if identity_sha256 != expected_identity:
        raise CorpusContractError(
            "calibration.identity_sha256 does not bind the canonical "
            "calibration descriptor"
        )
    out["identity_sha256"] = identity_sha256
    return out


def finalize_glm_bf16_corpus(
    *,
    incomplete_manifest_path: str | Path,
    source_artifact_path: str | Path,
    importance: Mapping[str, ImportanceVector],
    importance_identity: Mapping[str, object],
    output_artifact_path: str | Path,
    output_manifest_path: str | Path,
    calibration: Mapping[str, object],
    model_config_sha256: str,
    prismaquant_commit: str,
    generated: str,
    host: str,
) -> Path:
    """Finalize to new no-clobber paths, preserving all source weight bytes."""

    incomplete_path = Path(incomplete_manifest_path).resolve()
    source_path = Path(source_artifact_path).resolve()
    output_path = Path(output_artifact_path).resolve()
    manifest_path = Path(output_manifest_path).resolve()
    if output_path == source_path:
        raise CorpusContractError("finalizer never mutates the source artifact")
    if output_path.parent != manifest_path.parent:
        raise CorpusContractError("artifact and manifest must share one directory")
    if output_path.exists() or manifest_path.exists():
        raise CorpusContractError("finalized output paths are immutable and must not exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_contract = validate_incomplete_glm_source(incomplete_path, source_path)
    incomplete = _read_json_object(incomplete_path)
    raw_tensors = incomplete["tensors"]
    if set(importance) != set(raw_tensors):
        raise CorpusContractError("importance domain must exactly match source tensors")
    actual_source_file_hash = str(source_contract["file_sha256"])
    source_layout = _read_safetensors_layout(source_path)

    calibration_out = _canonical_calibration(calibration)
    config_hash = _require_sha256(model_config_sha256, where="model_config_sha256")
    if not _COMMIT_RE.fullmatch(str(prismaquant_commit)):
        raise CorpusContractError("prismaquant_commit must be a lowercase 40-hex commit")
    identity = dict(importance_identity)
    if identity.get("schema") != IMPORTANCE_SCHEMA:
        raise CorpusContractError("importance identity schema differs")
    if identity.get("probe_calibration_hash") != calibration_out["probe_calib_hash"]:
        raise CorpusContractError("importance calibration does not match corpus calibration")
    expected_value_hash = canonical_imatrix_sha256(
        {name: item.value for name, item in importance.items()}
    )
    if identity.get("value_sha256") != expected_value_hash:
        raise CorpusContractError("importance identity value hash differs")

    header: dict[str, object] = {}
    entries: list[dict[str, object]] = []
    for name, descriptor in sorted(
        source_layout.tensors.items(), key=lambda item: item[1]["data_offsets"][0]
    ):
        population, layer, projection, expert = _classify_glm_name(name)
        if population == "dense" and layer not in GLM_DENSE_LAYERS:
            raise CorpusContractError(f"{name}: unexpected dense layer")
        if population == "routed" and (layer not in GLM_ROUTED_LAYERS or expert != 0):
            raise CorpusContractError(f"{name}: unexpected routed layer/expert")
        shape = _expected_shape(population, projection)
        raw_meta = raw_tensors[name]
        if (
            descriptor.get("dtype") != "BF16"
            or tuple(descriptor.get("shape", ())) != shape
            or not isinstance(raw_meta, dict)
            or tuple(raw_meta.get("shape", ())) != shape
            or raw_meta.get("role") != population
        ):
            raise CorpusContractError(f"{name}: source dtype/shape/population differs")
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": list(descriptor["data_offsets"]),
        }

    cursor = source_layout.data_bytes
    importance_payloads: list[bytes] = []
    for name in sorted(raw_tensors):
        population, layer, projection, expert = _classify_glm_name(name)
        shape = _expected_shape(population, projection)
        vector = importance[name]
        if vector.target_name != name:
            raise CorpusContractError(f"{name}: importance target name differs")
        value = vector.value.detach().cpu().to(torch.float32).reshape(-1).contiguous()
        if (
            value.numel() != shape[1]
            or not bool(torch.isfinite(value).all())
            or bool((value < 0).any())
        ):
            raise CorpusContractError(f"{name}: invalid importance value")
        payload = _encode_f32(value)
        importance_key = f"__bf16_importance__.{name}"
        header[importance_key] = {
            "dtype": "F32",
            "shape": [shape[1]],
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
        importance_payloads.append(payload)
        weight_raw = _tensor_bytes(source_layout, name)
        entries.append({
            "name": name,
            "population": population,
            "layer": layer,
            "projection": projection,
            "expert": expert,
            "source_weight_dtype": "torch.bfloat16",
            "source_weight_shape": list(shape),
            "source_weight_sha256": hashlib.sha256(weight_raw).hexdigest(),
            "importance_key": importance_key,
            "importance_shape": [shape[1]],
            "importance_dtype": "torch.float32",
            "importance_sha256": hashlib.sha256(payload).hexdigest(),
            "importance_source": {
                "qname": vector.source_qname,
                "expert": vector.source_expert,
                "denominator_name": vector.denominator_name,
                "denominator": vector.denominator,
            },
            "census": {
                "distinct_source_values": raw_tensors[name].get("distinct_source_values"),
                "numel": math.prod(shape),
            },
        })

    header_bytes = _safetensors_header_bytes(header)
    artifact_tmp = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.partial-{os.getpid()}")
    if artifact_tmp.exists() or manifest_tmp.exists():
        raise CorpusContractError("stale finalizer temporary path exists")
    try:
        descriptor = os.open(artifact_tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            _write_all(descriptor, len(header_bytes).to_bytes(8, "little"))
            _write_all(descriptor, header_bytes)
            _copy_payload_pread(source_layout, descriptor)
            for payload in importance_payloads:
                _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        artifact_hash = _sha256_file(artifact_tmp)
        manifest: dict[str, object] = {
            "schema": FINALIZED_SCHEMA,
            "status": "finalized",
            "generated": str(generated),
            "host": str(host),
            "corpus_label": "GLM-5.3-Flash routed expert 0 + dense MLP, BF16 source",
            "model_profile": GLM_MODEL_PROFILE,
            "model": incomplete.get("model"),
            "model_config_sha256": config_hash,
            "num_hidden_layers": GLM_NUM_HIDDEN_LAYERS,
            "layers": list(GLM_LAYERS),
            "roles": list(GLM_PROJECTIONS),
            "expert": GLM_EXPERT,
            "calibration": calibration_out,
            "importance_identity": identity,
            "reader_contract": {
                "schema": READER_SCHEMA,
                "method": "os.pread",
                "mmap": False,
                "source_weight_dtype": "torch.bfloat16",
                "importance_dtype": "torch.float32",
                "consumer_weight_dtype": "torch.float32",
            },
            "prismaquant_commit": str(prismaquant_commit),
            "file": output_path.name,
            "file_size_bytes": artifact_tmp.stat().st_size,
            "file_sha256": artifact_hash,
            "populations": {
                "dense": {"count": GLM_DENSE_COUNT, "layers": list(GLM_DENSE_LAYERS)},
                "routed": {"count": GLM_ROUTED_COUNT, "layers": list(GLM_ROUTED_LAYERS)},
            },
            "source_artifact": {
                "manifest_schema": INCOMPLETE_GLM_SCHEMA,
                "file_sha256": actual_source_file_hash,
                "payload_bytes": source_layout.data_bytes,
                "payload_copy": "byte-for-byte via os.pread",
            },
            "entries": entries,
        }
        # Validate the complete pair while both members still have temporary,
        # private names.  Only the basename differs from the final manifest;
        # no invalid artifact is ever published at the requested path.
        validation_manifest = {**manifest, "file": artifact_tmp.name}
        manifest_tmp.write_text(
            json.dumps(validation_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        load_finalized_bf16_corpus(manifest_tmp)
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with manifest_tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.rename(artifact_tmp, output_path)
        os.rename(manifest_tmp, manifest_path)
    except Exception:
        for temporary in (artifact_tmp, manifest_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise
    # Re-open through the strict consumer after the two final renames too.
    load_finalized_bf16_corpus(manifest_path)
    return manifest_path


__all__ = [
    "CorpusContractError",
    "CorpusEntry",
    "FinalizedBF16Corpus",
    "ImportanceVector",
    "FINALIZED_SCHEMA",
    "IMPORTANCE_SCHEMA",
    "INCOMPLETE_GLM_SCHEMA",
    "GLM_DENSE_COUNT",
    "GLM_ROUTED_COUNT",
    "adapt_glm_importance_from_probe",
    "finalize_glm_bf16_corpus",
    "load_finalized_bf16_corpus",
    "validate_incomplete_glm_source",
]
