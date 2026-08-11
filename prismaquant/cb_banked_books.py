"""Fail-closed loader for immutable CBL books measured by a burn campaign.

The bank is not a catalog.  Callers must name every accepted burn-cell shard
explicitly and supply the source/imatrix digests expected for that cell.  This
module follows the shard's content-addressed pointer, verifies the complete
identity chain, and returns the exact FP16 safetensors payload.  It never
searches a directory, trains a replacement book, or falls back to a lattice.

The persisted ``book_sha256`` is the historical pooled-Lloyd digest: the
concatenation of the subtables after promotion to float32.  Because the learned
grid values round-trip through FP16 exactly, it is recomputed from the stored
payload in addition to the per-subtable FP16 payload digests exposed here.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import re
from typing import Any

import torch
from safetensors import safe_open

from .cb_layout import codebook_subtable_shapes, parse_format_name
from .routed_moe_codebooks import ROUTED_MOE_CBL_BANK_RUNGS


BURN_CELL_SCHEMA = "prismaquant.dsv4_afast_burn_cell.v4"
BURN_CELL_IDENTITY_SCHEMA = "prismaquant.dsv4_afast_burn_cell_identity.v4"
BURN_PASS_TAG_SCHEMA = "prismaquant.dsv4_afast_burn_pass_tags.v1"
BANKED_CBL_BOOK_SCHEMA = "prismaquant.dsv4_cbl_book.v1"
BANKED_CBL_BOOK_METADATA_KEY = "dsv4_cbl_book"
BANKED_CBL_ORIGIN_SCHEMA = "prismaquant.routed_moe_banked_cbl_origin.v1"
ROUTED_MOE_CBL_SELECTION_SCHEMA = (
    "prismaquant.routed_moe_cbl_book_selection.v1"
)

# This is the stamp stored by the completed DSv4 burn.  Its field names differ
# intentionally from the production bundle's normalized trainer stamp.
BANKED_CBL_TRAIN_STAMP: dict[str, object] = {
    "row_sample": 64,
    "row_seed": 4321,
    "cap": 2_000_000,
    "iters": 4,
    "seed": 0,
    "init": "fixed_lattice",
    "normalization": "cand0_v1",
}

_PROJECTIONS = frozenset({"gate_proj", "up_proj", "down_proj"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BANKED_ORIGIN_MEMBERS = {
    "schema",
    "selection_sha256",
    "selection_path",
    "burn_shard",
    "burn_content_key",
    "burn_pass_tag",
    "book_path",
    "book_key",
    "book_file_sha256",
    "pooled_book_sha256",
    "subtable_content_sha256",
    "layer",
    "projection",
    "rung",
    "source_digest",
    "col_weights_digest",
}


class BankedCBLBookError(ValueError):
    """An accepted burn shard or its referenced book is stale or corrupt."""


@dataclass(frozen=True)
class RoutedMoECBLSelectionCell:
    """One operator-accepted routed role/rung burn result."""

    layer: int
    projection: str
    rung: int
    burn_shard_path: Path

    @property
    def format_name(self) -> str:
        return f"FP8_CBL_K{self.rung}"


@dataclass(frozen=True)
class RoutedMoECBLSelection:
    """Strict, immutable input selecting exact banked routed books."""

    path: Path
    content_sha256: str
    book_root: Path
    cells: tuple[RoutedMoECBLSelectionCell, ...]

    @property
    def by_cell(self) -> dict[tuple[int, str, int], RoutedMoECBLSelectionCell]:
        return {
            (cell.layer, cell.projection, cell.rung): cell
            for cell in self.cells
        }


@dataclass(frozen=True)
class BankedCBLBookRequest:
    """One caller-selected burn cell and the identities it must match."""

    burn_shard_path: Path
    layer: int
    projection: str
    rung: int
    source_digest: str
    col_weights_digest: str
    source_shape: tuple[int, ...] | None = None
    col_weights_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class BankedCBLBook:
    """Verified exact-byte snapshot of one pooled-Lloyd expert book."""

    burn_shard_path: Path
    book_path: Path
    pass_tag: str
    layer: int
    projection: str
    rung: int
    source_shape: tuple[int, ...]
    source_digest: str
    col_weights_shape: tuple[int, ...]
    col_weights_digest: str
    encoded_expert_ids: tuple[int, ...]
    cbl_semantics_schema: str
    burn_content_key: str
    book_key: str
    book_sha256: str
    book_file_sha256: str
    subtable_content_sha256: tuple[str, ...]
    subtables: tuple[torch.Tensor, ...]

    @property
    def format_name(self) -> str:
        return f"FP8_CBL_K{self.rung}"


def banked_cbl_origin(
    selection: RoutedMoECBLSelection,
    book: BankedCBLBook,
) -> dict[str, object]:
    """Auditable bundle-cell provenance for one verified bank snapshot."""

    if not isinstance(selection, RoutedMoECBLSelection):
        raise TypeError("selection must be a RoutedMoECBLSelection")
    if not isinstance(book, BankedCBLBook):
        raise TypeError("book must be a BankedCBLBook")
    return {
        "schema": BANKED_CBL_ORIGIN_SCHEMA,
        "selection_sha256": selection.content_sha256,
        "selection_path": str(selection.path),
        "burn_shard": str(book.burn_shard_path),
        "burn_content_key": book.burn_content_key,
        "burn_pass_tag": book.pass_tag,
        "book_path": str(book.book_path),
        "book_key": book.book_key,
        "book_file_sha256": book.book_file_sha256,
        "pooled_book_sha256": book.book_sha256,
        "subtable_content_sha256": list(book.subtable_content_sha256),
        "layer": book.layer,
        "projection": book.projection,
        "rung": book.rung,
        "source_digest": book.source_digest,
        "col_weights_digest": book.col_weights_digest,
    }


def validate_banked_cbl_origin(
    value: object, *, where: str = "banked CBL origin"
) -> dict[str, object]:
    """Validate persisted bank provenance without reopening campaign files."""

    origin = _mapping(value, where=where)
    if set(origin) != _BANKED_ORIGIN_MEMBERS:
        raise BankedCBLBookError(f"{where}: origin members differ")
    if origin.get("schema") != BANKED_CBL_ORIGIN_SCHEMA:
        raise BankedCBLBookError(f"{where}: unsupported origin schema")
    for member in (
        "selection_sha256",
        "burn_content_key",
        "book_key",
        "book_file_sha256",
        "pooled_book_sha256",
        "source_digest",
        "col_weights_digest",
    ):
        _digest(origin.get(member), where=f"{where} {member}")
    for member in ("selection_path", "burn_shard", "book_path"):
        path = origin.get(member)
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise BankedCBLBookError(
                f"{where} {member}: expected an absolute path"
            )
    pass_tag = origin.get("burn_pass_tag")
    if not isinstance(pass_tag, str) or not pass_tag:
        raise BankedCBLBookError(f"{where}: missing burn pass tag")
    layer = _strict_int(origin.get("layer"), where=f"{where} layer")
    rung = _strict_int(origin.get("rung"), where=f"{where} rung")
    if layer < 0 or rung not in ROUTED_MOE_CBL_BANK_RUNGS:
        raise BankedCBLBookError(f"{where}: invalid routed cell coordinates")
    if origin.get("projection") not in _PROJECTIONS:
        raise BankedCBLBookError(f"{where}: invalid routed projection")
    digests = origin.get("subtable_content_sha256")
    if not isinstance(digests, list) or not digests:
        raise BankedCBLBookError(f"{where}: missing subtable digests")
    for index, digest in enumerate(digests):
        _digest(digest, where=f"{where} subtable digest {index}")
    return dict(origin)


def _mapping(value: object, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BankedCBLBookError(f"{where}: expected an object")
    return value


def _strict_int(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BankedCBLBookError(f"{where}: expected an integer")
    return int(value)


def _digest(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BankedCBLBookError(
            f"{where}: expected a lowercase 64-hex SHA-256"
        )
    return value


def _shape(value: object, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise BankedCBLBookError(f"{where}: expected a non-empty shape")
    result = tuple(_strict_int(dim, where=f"{where}[{index}]") for index, dim in enumerate(value))
    if any(dim <= 0 for dim in result):
        raise BankedCBLBookError(f"{where}: dimensions must be positive")
    return result


def _expert_ids(value: object, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise BankedCBLBookError(f"{where}: expected a non-empty expert-id list")
    result = tuple(
        _strict_int(expert, where=f"{where}[{index}]")
        for index, expert in enumerate(value)
    )
    if any(expert < 0 for expert in result) or len(set(result)) != len(result):
        raise BankedCBLBookError(
            f"{where}: expert ids must be unique non-negative integers"
        )
    return result


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BankedCBLBookError(
            f"burn identity is not canonical JSON data: {exc}"
        ) from exc


def _strict_json_loads(raw: object, *, where: str) -> object:
    if not isinstance(raw, str):
        raise BankedCBLBookError(f"{where}: metadata value must be JSON text")

    def reject_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise BankedCBLBookError(
                    f"{where}: duplicate JSON member {key!r}"
                )
            out[key] = value
        return out

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise BankedCBLBookError(f"{where}: malformed JSON: {exc}") from exc


def load_routed_moe_cbl_selection(
    path: str | Path,
) -> RoutedMoECBLSelection:
    """Load an explicit routed-book selection without searching the bank.

    The JSON contains only absolute paths: one bank root and one accepted
    burn shard for each ``(layer, projection, rung)`` cell.  Source and
    imatrix digests deliberately do not live in this operator selection;
    :func:`load_banked_cbl_book` compares the selected shard against the
    producer's current tensors when the bundle is built.
    """

    selection_path = Path(path)
    if not selection_path.is_file():
        raise BankedCBLBookError(
            f"{selection_path}: routed-MoE CBL selection is missing"
        )
    try:
        raw = selection_path.read_bytes()
    except OSError as exc:
        raise BankedCBLBookError(
            f"{selection_path}: routed-MoE CBL selection is unreadable: {exc}"
        ) from exc
    payload = _mapping(
        _strict_json_loads(
            raw.decode("utf-8"), where=str(selection_path)
        ),
        where=str(selection_path),
    )
    if set(payload) != {"schema", "book_root", "cells"}:
        raise BankedCBLBookError(
            f"{selection_path}: selection members differ; expected exactly "
            "schema, book_root, cells"
        )
    if payload.get("schema") != ROUTED_MOE_CBL_SELECTION_SCHEMA:
        raise BankedCBLBookError(
            f"{selection_path}: unsupported routed-book selection schema "
            f"{payload.get('schema')!r}"
        )
    raw_root = payload.get("book_root")
    if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
        raise BankedCBLBookError(
            f"{selection_path}: book_root must be an absolute path"
        )
    book_root = Path(raw_root).resolve(strict=False)
    if not book_root.is_dir():
        raise BankedCBLBookError(
            f"{book_root}: selected book_root is not a directory"
        )
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise BankedCBLBookError(
            f"{selection_path}: cells must be a non-empty array"
        )
    cells: list[RoutedMoECBLSelectionCell] = []
    seen: set[tuple[int, str, int]] = set()
    for index, raw_cell in enumerate(raw_cells):
        where = f"{selection_path} cells[{index}]"
        cell = _mapping(raw_cell, where=where)
        if set(cell) != {"layer", "projection", "rung", "burn_shard"}:
            raise BankedCBLBookError(
                f"{where}: members differ; expected exactly layer, "
                "projection, rung, burn_shard"
            )
        layer = _strict_int(cell.get("layer"), where=f"{where} layer")
        if layer < 0:
            raise BankedCBLBookError(f"{where}: layer must be non-negative")
        projection = cell.get("projection")
        if projection not in _PROJECTIONS:
            raise BankedCBLBookError(
                f"{where}: projection must be one of {sorted(_PROJECTIONS)}"
            )
        rung = _strict_int(cell.get("rung"), where=f"{where} rung")
        if rung not in ROUTED_MOE_CBL_BANK_RUNGS:
            raise BankedCBLBookError(
                f"{where}: routed-MoE bank selections support K28-K33 only; "
                f"got K{rung}"
            )
        raw_shard = cell.get("burn_shard")
        if not isinstance(raw_shard, str) or not Path(raw_shard).is_absolute():
            raise BankedCBLBookError(
                f"{where}: burn_shard must be an absolute path"
            )
        shard = Path(raw_shard).resolve(strict=False)
        if not shard.is_file():
            raise BankedCBLBookError(
                f"{shard}: accepted burn shard is missing; refusing scan or "
                "fallback"
            )
        key = (layer, str(projection), rung)
        if key in seen:
            raise BankedCBLBookError(
                f"{selection_path}: duplicate selected cell L{layer} "
                f"{projection} K{rung}"
            )
        seen.add(key)
        cells.append(RoutedMoECBLSelectionCell(
            layer=layer,
            projection=str(projection),
            rung=rung,
            burn_shard_path=shard,
        ))
    cells.sort(key=lambda cell: (
        cell.layer, cell.projection, cell.rung,
    ))
    return RoutedMoECBLSelection(
        path=selection_path.resolve(strict=True),
        content_sha256=hashlib.sha256(raw).hexdigest(),
        book_root=book_root,
        cells=tuple(cells),
    )


def _same_json(left: object, right: object) -> bool:
    """JSON equality that does not treat booleans as integers."""

    return _canonical_json(left) == _canonical_json(right)


def _burn_content_key(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _legacy_book_key(
    *,
    semantics_schema: str,
    layer: int,
    projection: str,
    rung: int,
    source_digest: str,
    col_weights_digest: str,
    train: Mapping[str, Any],
) -> str:
    # Match tools.dsv4_cbl_kernels._book_key byte-for-byte.  That historical
    # key intentionally used json.dumps' default separators.
    payload = json.dumps(
        {
            "schema": semantics_schema,
            "layer": layer,
            "projection": projection,
            "rung": rung,
            "source_digest": source_digest,
            "col_weights_digest": col_weights_digest,
            "train": dict(train),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _fp16_payload_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.numpy().astype("<f2", copy=False).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _historical_pool_sha256(subtables: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for table in subtables:
        payload = (
            table.to(torch.float32)
            .numpy()
            .astype("<f4", copy=False)
            .tobytes(order="C")
        )
        digest.update(payload)
    return digest.hexdigest()


def _load_burn_shard(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise BankedCBLBookError(
            f"{path}: accepted burn shard is missing; refusing bank fallback"
        )
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise BankedCBLBookError(
            f"{path}: accepted burn shard is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return _mapping(payload, where=str(path))


def load_banked_cbl_book(
    request: BankedCBLBookRequest,
    *,
    book_root: str | Path,
) -> BankedCBLBook:
    """Resolve one explicitly accepted burn shard to exact FP16 subtables.

    ``source_digest`` and ``col_weights_digest`` should be computed from the
    producer's current role-specific rank-3 tensors with
    :func:`prismaquant.cb_warm_state.tensor_value_identity`.  A mismatch is a
    hard error; this function has no training or lattice code path.
    """

    if not isinstance(request, BankedCBLBookRequest):
        raise TypeError("request must be a BankedCBLBookRequest")
    shard_path = Path(request.burn_shard_path).resolve(strict=False)
    expected_layer = _strict_int(request.layer, where="expected layer")
    expected_rung = _strict_int(request.rung, where="expected rung")
    expected_projection = str(request.projection)
    if expected_projection not in _PROJECTIONS:
        raise BankedCBLBookError(
            f"expected projection {expected_projection!r} is not one of "
            f"{sorted(_PROJECTIONS)}"
        )
    expected_source_digest = _digest(
        request.source_digest, where="expected source digest"
    )
    expected_col_digest = _digest(
        request.col_weights_digest, where="expected col_weights digest"
    )
    expected_source_shape = (
        None
        if request.source_shape is None
        else _shape(request.source_shape, where="expected source shape")
    )
    expected_col_shape = (
        None
        if request.col_weights_shape is None
        else _shape(request.col_weights_shape, where="expected col_weights shape")
    )

    parsed = parse_format_name(f"FP8_CBL_K{expected_rung}")
    if (
        parsed is None
        or parsed[0].grid != "fp8"
        or parsed[0].mode != "product"
        or parsed[0].source != "learned"
    ):
        raise BankedCBLBookError(
            f"K{expected_rung}: not a structural FP8 product-codebook rung"
        )
    family = parsed[0]

    payload = _load_burn_shard(shard_path)
    if payload.get("schema") != BURN_CELL_SCHEMA:
        raise BankedCBLBookError(
            f"{shard_path}: unsupported burn-cell schema {payload.get('schema')!r}"
        )
    if payload.get("pass_tag_schema") != BURN_PASS_TAG_SCHEMA:
        raise BankedCBLBookError(f"{shard_path}: burn pass-tag schema differs")
    pass_tag = payload.get("pass_tag")
    if not isinstance(pass_tag, str) or not pass_tag:
        raise BankedCBLBookError(f"{shard_path}: missing burn pass tag")

    identity = _mapping(payload.get("identity"), where=f"{shard_path} identity")
    if identity.get("schema") != BURN_CELL_IDENTITY_SCHEMA:
        raise BankedCBLBookError(f"{shard_path}: burn-cell identity schema differs")
    if (
        identity.get("pass_tag_schema") != BURN_PASS_TAG_SCHEMA
        or identity.get("pass_tag") != pass_tag
    ):
        raise BankedCBLBookError(f"{shard_path}: pass tag identity differs")
    content_key = _digest(payload.get("content_key"), where=f"{shard_path} content_key")
    observed_content_key = _burn_content_key(identity)
    if content_key != observed_content_key:
        raise BankedCBLBookError(
            f"{shard_path}: burn identity content-key mismatch"
        )

    layer = _strict_int(identity.get("layer"), where=f"{shard_path} layer")
    projection = identity.get("projection")
    rung = _strict_int(identity.get("rung"), where=f"{shard_path} rung")
    if (layer, projection, rung) != (
        expected_layer,
        expected_projection,
        expected_rung,
    ):
        raise BankedCBLBookError(
            f"{shard_path}: burn cell {(layer, projection, rung)!r} != "
            f"requested {(expected_layer, expected_projection, expected_rung)!r}"
        )

    expert_ids = _expert_ids(
        identity.get("expert_ids"), where=f"{shard_path} expert_ids"
    )
    encoded_expert_ids = _expert_ids(
        identity.get("encoded_expert_ids"),
        where=f"{shard_path} encoded_expert_ids",
    )
    guard = _mapping(
        identity.get("content_guard"), where=f"{shard_path} content_guard"
    )
    source_shape = _shape(
        guard.get("source_shape"), where=f"{shard_path} source_shape"
    )
    col_shape = _shape(
        guard.get("col_weights_shape"), where=f"{shard_path} col_weights_shape"
    )
    source_digest = _digest(
        guard.get("source_digest"), where=f"{shard_path} source_digest"
    )
    col_digest = _digest(
        guard.get("col_weights_digest"),
        where=f"{shard_path} col_weights_digest",
    )
    if source_digest != expected_source_digest:
        raise BankedCBLBookError(
            f"{shard_path}: source digest differs from requested role tensor"
        )
    if col_digest != expected_col_digest:
        raise BankedCBLBookError(
            f"{shard_path}: col_weights digest differs from requested role tensor"
        )
    if expected_source_shape is not None and source_shape != expected_source_shape:
        raise BankedCBLBookError(f"{shard_path}: source shape differs from requested role tensor")
    if expected_col_shape is not None and col_shape != expected_col_shape:
        raise BankedCBLBookError(
            f"{shard_path}: col_weights shape differs from requested role tensor"
        )
    if source_shape[0] != len(encoded_expert_ids):
        raise BankedCBLBookError(
            f"{shard_path}: encoded expert population does not match source shape"
        )
    if col_shape[0] != len(encoded_expert_ids):
        raise BankedCBLBookError(
            f"{shard_path}: encoded expert population does not match col_weights shape"
        )
    aggregate_col_digest = guard.get("col_weights_sha256")
    if aggregate_col_digest is not None and _digest(
        aggregate_col_digest, where=f"{shard_path} col_weights_sha256"
    ) != col_digest:
        raise BankedCBLBookError(
            f"{shard_path}: aggregate and role col_weights digests differ"
        )

    semantics = _mapping(
        guard.get("cbl_semantics"), where=f"{shard_path} cbl_semantics"
    )
    semantics_schema = semantics.get("schema")
    if not isinstance(semantics_schema, str) or not semantics_schema:
        raise BankedCBLBookError(f"{shard_path}: missing CBL semantics schema")
    if semantics.get("adopted_encoder") != "cbl_poolb":
        raise BankedCBLBookError(
            f"{shard_path}: CBL semantics did not adopt cbl_poolb"
        )
    if semantics.get("ldlq_in_measurement") is not False:
        raise BankedCBLBookError(
            f"{shard_path}: banked CBL must have ldlq_in_measurement=false"
        )
    train = _mapping(
        semantics.get("book_train"), where=f"{shard_path} book_train"
    )
    if not _same_json(train, BANKED_CBL_TRAIN_STAMP):
        raise BankedCBLBookError(
            f"{shard_path}: pooled-Lloyd trainer identity differs"
        )

    cell = _mapping(payload.get("cell"), where=f"{shard_path} cell")
    if (
        _strict_int(cell.get("rung"), where=f"{shard_path} cell rung") != rung
        or cell.get("pass_tag_schema") != BURN_PASS_TAG_SCHEMA
        or cell.get("pass_tag") != pass_tag
    ):
        raise BankedCBLBookError(f"{shard_path}: cell identity differs")
    if _expert_ids(cell.get("expert_ids"), where=f"{shard_path} cell expert_ids") != expert_ids:
        raise BankedCBLBookError(f"{shard_path}: cell expert set differs")
    if _expert_ids(
        cell.get("encoded_expert_ids"),
        where=f"{shard_path} cell encoded_expert_ids",
    ) != encoded_expert_ids:
        raise BankedCBLBookError(f"{shard_path}: cell encoded expert set differs")
    timing = _mapping(cell.get("timing"), where=f"{shard_path} timing")
    measurement = _mapping(
        timing.get("measurement_semantics"),
        where=f"{shard_path} measurement_semantics",
    )
    if (
        measurement.get("schema") != semantics_schema
        or measurement.get("encoder") != "cbl_poolb"
        or measurement.get("ldlq") is not False
        or measurement.get("scale_policy") != "one_shot_cand0"
    ):
        raise BankedCBLBookError(
            f"{shard_path}: measurement is not the certified cbl_poolb arm"
        )
    book_sha256 = _digest(
        measurement.get("book_sha256"), where=f"{shard_path} book_sha256"
    )

    book_key = _legacy_book_key(
        semantics_schema=semantics_schema,
        layer=layer,
        projection=projection,
        rung=rung,
        source_digest=source_digest,
        col_weights_digest=col_digest,
        train=train,
    )
    root = Path(book_root).resolve(strict=True)
    if not root.is_dir():
        raise BankedCBLBookError(f"{root}: book_root is not a directory")
    expected_book_path = root / book_key[:2] / f"{book_key}.safetensors"
    raw_book_path = measurement.get("book_path")
    if not isinstance(raw_book_path, str) or not Path(raw_book_path).is_absolute():
        raise BankedCBLBookError(
            f"{shard_path}: book_path must be an absolute content-addressed path"
        )
    book_path = Path(raw_book_path).resolve(strict=False)
    if book_path != expected_book_path.resolve(strict=False):
        raise BankedCBLBookError(
            f"{shard_path}: book path does not match its content-addressed key"
        )
    if cell.get("warm_state_path") != raw_book_path:
        raise BankedCBLBookError(
            f"{shard_path}: warm-state path differs from measurement book path"
        )
    if not book_path.is_file():
        raise BankedCBLBookError(
            f"{book_path}: accepted banked book is missing; refusing retraining "
            "or lattice fallback"
        )

    try:
        before = _file_identity(book_path)
        book_file_sha256 = _file_sha256(book_path)
        with safe_open(str(book_path), framework="pt", device="cpu") as handle:
            raw_metadata = handle.metadata() or {}
            if set(raw_metadata) != {BANKED_CBL_BOOK_METADATA_KEY}:
                raise BankedCBLBookError(
                    f"{book_path}: banked-book metadata members differ"
                )
            book_metadata = _strict_json_loads(
                raw_metadata[BANKED_CBL_BOOK_METADATA_KEY], where=str(book_path)
            )
            metadata = _mapping(book_metadata, where=f"{book_path} metadata")
            expected_metadata_members = {
                "schema",
                "book_key",
                "book_sha256",
                "n_sub",
                "layer",
                "projection",
                "rung",
                "source_digest",
                "col_weights_digest",
                "train",
                "device_class",
            }
            if set(metadata) != expected_metadata_members:
                raise BankedCBLBookError(
                    f"{book_path}: banked-book manifest members differ: "
                    f"missing={sorted(expected_metadata_members - set(metadata))}, "
                    f"unknown={sorted(set(metadata) - expected_metadata_members)}"
                )
            n_sub = _strict_int(metadata.get("n_sub"), where=f"{book_path} n_sub")
            if n_sub != family.n_sub:
                raise BankedCBLBookError(
                    f"{book_path}: n_sub={n_sub} != FP8 product ABI {family.n_sub}"
                )
            expected_names = tuple(f"sub{index}" for index in range(n_sub))
            if set(handle.keys()) != set(expected_names):
                raise BankedCBLBookError(
                    f"{book_path}: subtable tensor-name set differs"
                )
            subtables = tuple(
                handle.get_tensor(name).detach().clone().contiguous()
                for name in expected_names
            )
        after = _file_identity(book_path)
    except BankedCBLBookError:
        raise
    except Exception as exc:
        raise BankedCBLBookError(
            f"{book_path}: banked book is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if before != after:
        raise BankedCBLBookError(f"{book_path}: banked book changed while loading")

    expected_shapes = codebook_subtable_shapes(rung, family.mode, family.n_sub)
    if len(subtables) != len(expected_shapes):
        raise BankedCBLBookError(
            f"{book_path}: expected {len(expected_shapes)} subtables, got {len(subtables)}"
        )
    for index, (table, expected_shape) in enumerate(
        zip(subtables, expected_shapes, strict=True)
    ):
        if table.dtype != torch.float16 or table.ndim != 2:
            raise BankedCBLBookError(
                f"{book_path}: sub{index} must be rank-2 FP16, got "
                f"{tuple(table.shape)} {table.dtype}"
            )
        if tuple(int(dim) for dim in table.shape) != tuple(expected_shape):
            raise BankedCBLBookError(
                f"{book_path}: sub{index} shape {tuple(table.shape)} != "
                f"{tuple(expected_shape)}"
            )
        if not bool(torch.isfinite(table).all()):
            raise BankedCBLBookError(f"{book_path}: sub{index} contains non-finite values")

    if metadata.get("schema") != BANKED_CBL_BOOK_SCHEMA:
        raise BankedCBLBookError(f"{book_path}: banked-book schema differs")
    metadata_key = _digest(metadata.get("book_key"), where=f"{book_path} book_key")
    metadata_sha = _digest(
        metadata.get("book_sha256"), where=f"{book_path} book_sha256"
    )
    metadata_train = _mapping(
        metadata.get("train"), where=f"{book_path} train"
    )
    if (
        metadata_key != book_key
        or book_path.stem != book_key
        or book_path.parent.name != book_key[:2]
    ):
        raise BankedCBLBookError(f"{book_path}: banked-book key identity differs")
    if metadata_sha != book_sha256:
        raise BankedCBLBookError(
            f"{book_path}: burn shard and book metadata hashes differ"
        )
    if not _same_json(metadata_train, BANKED_CBL_TRAIN_STAMP):
        raise BankedCBLBookError(f"{book_path}: book trainer identity differs")
    metadata_identity = (
        _strict_int(metadata.get("layer"), where=f"{book_path} layer"),
        metadata.get("projection"),
        _strict_int(metadata.get("rung"), where=f"{book_path} rung"),
        _digest(metadata.get("source_digest"), where=f"{book_path} source_digest"),
        _digest(
            metadata.get("col_weights_digest"),
            where=f"{book_path} col_weights_digest",
        ),
    )
    if metadata_identity != (layer, projection, rung, source_digest, col_digest):
        raise BankedCBLBookError(f"{book_path}: book cell/source identity differs")

    observed_pool_sha = _historical_pool_sha256(subtables)
    if observed_pool_sha != book_sha256:
        raise BankedCBLBookError(
            f"{book_path}: stored FP16 subtables do not reproduce book_sha256"
        )
    subtable_digests = tuple(_fp16_payload_sha256(table) for table in subtables)

    return BankedCBLBook(
        burn_shard_path=shard_path,
        book_path=book_path,
        pass_tag=pass_tag,
        layer=layer,
        projection=projection,
        rung=rung,
        source_shape=source_shape,
        source_digest=source_digest,
        col_weights_shape=col_shape,
        col_weights_digest=col_digest,
        encoded_expert_ids=encoded_expert_ids,
        cbl_semantics_schema=semantics_schema,
        burn_content_key=content_key,
        book_key=book_key,
        book_sha256=book_sha256,
        book_file_sha256=book_file_sha256,
        subtable_content_sha256=subtable_digests,
        subtables=subtables,
    )


def load_banked_cbl_books(
    requests: Iterable[BankedCBLBookRequest],
    *,
    book_root: str | Path,
) -> dict[tuple[int, str, int], BankedCBLBook]:
    """Load an explicit accepted set; duplicate cells fail closed."""

    result: dict[tuple[int, str, int], BankedCBLBook] = {}
    for request in requests:
        book = load_banked_cbl_book(request, book_root=book_root)
        key = (book.layer, book.projection, book.rung)
        if key in result:
            raise BankedCBLBookError(
                f"duplicate accepted burn shard for L{book.layer} "
                f"{book.projection} K{book.rung}"
            )
        result[key] = book
    if not result:
        raise BankedCBLBookError("no accepted burn shards were supplied")
    return result
