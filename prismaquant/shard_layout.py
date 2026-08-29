"""HF-standard safetensors shard layout, shared by the export lanes.

Every PrismaQuant lane publishes the same layout a stock HF/vLLM loader
expects, so that "which exporter wrote this?" is never a question a consumer
has to answer:

* one resulting shard -> ``model.safetensors`` and **no** index file;
* N > 1 -> ``model-00001-of-000NN.safetensors`` plus
  ``model.safetensors.index.json``.

The partition rule is the one the compressed-tensors lane already ships
(``export_native_compressed.IncrementalSafetensorsWriter.add_tensors``):
accumulate in emit order, close the current shard when the next tensor would
push it past the budget, and give a tensor larger than the whole budget its
own shard rather than splitting it across files. Emit order is preserved, so
the same tensor sequence and the same budget always produce the same layout.

There is deliberately no "0 means one file" sentinel: the native lane has
none, and inventing one here would make the two lanes disagree about the same
flag. The single-file layout is what a budget at least as large as the
artifact already produces, which is how every pre-2026-08-21 CB artifact is
reproduced.

Why 1 GiB: a GB10 user loading the 87 GB single-container DSv4 CB artifact
stalled the default HF loader on a 128 GB unified-memory box and resharded it
by hand (RobTand/gridbook#47). ``scripts/reshard_safetensors.py`` is the
after-the-fact repair this module makes unnecessary at the export boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

__all__ = [
    "DEFAULT_SHARD_BYTES",
    "SHARD_INDEX_NAME",
    "SHARD_NAME_RE",
    "SINGLE_CONTAINER_NAME",
    "TENSOR_PAYLOAD_IDENTITY_SCHEMA",
    "container_names",
    "describe_container_layout",
    "plan_shards",
    "shard_name",
    "tensor_payload_identity",
    "write_shard_index",
]

# Robert's standing packaging default since 2026-08-20; the native lane spells
# the same number at `export_native_compressed`'s `--shard-bytes` default and
# `run-pipeline.sh`'s `EXPORT_SHARD_BYTES`.
DEFAULT_SHARD_BYTES = 1024 ** 3

SINGLE_CONTAINER_NAME = "model.safetensors"
SHARD_INDEX_NAME = "model.safetensors.index.json"
SHARD_NAME_RE = re.compile(r"^model-([0-9]{5})-of-([0-9]{5})\.safetensors$")

TENSOR_PAYLOAD_IDENTITY_SCHEMA = "prismaquant.tensor_payload_identity/1"


def tensor_payload_identity(
    tensor_sha256: Mapping[str, str],
    *,
    include_tensor_sha256: bool = False,
) -> dict[str, object]:
    """Reduce per-tensor content digests to one layout-invariant identity.

    ``shipcard.compute_model_sha`` binds the *container filenames and sizes*
    (``shipcard.py:270-279``), so a model_sha necessarily moves when the shard
    budget changes -- it is a statement about the published files. This digest
    is over the tensor payload alone (name -> sha256 of the tensor's raw
    bytes), so it does **not** move, which is what makes a reshard of identical
    tensors recognisable as the same model.

    Both CB exporters stamp it at ``provenance.tensor_payload_identity``, from
    digests taken in the pass that already touches the bytes.
    """
    if not tensor_sha256:
        raise ValueError("cannot identify an empty tensor payload")
    rows = {str(name): str(digest) for name, digest in tensor_sha256.items()}
    if (
        any(not name for name in rows)
        or len(rows) != len(tensor_sha256)
        or any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in rows.values())
    ):
        raise ValueError(
            "tensor payload identity requires unique nonempty names and "
            "lowercase SHA-256 content digests"
        )
    result: dict[str, object] = {
        "schema": TENSOR_PAYLOAD_IDENTITY_SCHEMA,
        "algorithm": "sha256",
        "tensors": len(rows),
        "payload_sha256": hashlib.sha256(json.dumps(
            dict(sorted(rows.items())),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest(),
    }
    if include_tensor_sha256:
        # Strict release artifacts retain the producer's already-computed
        # per-tensor ledger so the aggregate can be replayed against the exact
        # finalized header census without rereading an 18 GB payload.  Generic
        # artifacts keep the compact historical wire shape by default.
        result["tensor_sha256"] = dict(sorted(rows.items()))
    return result


def shard_name(index: int, count: int) -> str:
    """``model-00003-of-00024.safetensors`` for the 1-based ``index``."""
    if count < 1:
        raise ValueError(f"shard count must be positive: {count}")
    if not 1 <= index <= count:
        raise ValueError(f"shard index {index} outside 1..{count}")
    return f"model-{index:05d}-of-{count:05d}.safetensors"


def container_names(count: int) -> list[str]:
    """The published weight-container filenames for ``count`` shards."""
    if count < 1:
        raise ValueError(f"shard count must be positive: {count}")
    if count == 1:
        return [SINGLE_CONTAINER_NAME]
    return [shard_name(i, count) for i in range(1, count + 1)]


def plan_shards(
    sizes: Sequence[tuple[str, int]],
    shard_bytes: int,
) -> list[list[str]]:
    """Partition ``(name, nbytes)`` in emit order into shard name groups.

    Deterministic by construction: the result is a pure function of the input
    sequence and the budget.
    """
    budget = int(shard_bytes)
    if budget <= 0:
        raise ValueError(
            "shard_bytes must be positive; there is no zero sentinel. To "
            "publish the legacy single-container layout, pass a budget at "
            "least as large as the finished artifact."
        )
    if not sizes:
        raise ValueError("cannot plan shards for an empty tensor set")

    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name, nbytes in sizes:
        size = int(nbytes)
        if size < 0:
            raise ValueError(f"{name}: negative tensor size {size}")
        if current and current_size + size > budget:
            shards.append(current)
            current = []
            current_size = 0
        current.append(str(name))
        current_size += size
        # A single tensor may exceed the whole budget. It gets its own shard
        # rather than being split: safetensors has no cross-file tensor.
        if current_size >= budget:
            shards.append(current)
            current = []
            current_size = 0
    if current:
        shards.append(current)
    return shards


def write_shard_index(
    out_dir: str | Path,
    weight_map: Mapping[str, str],
    total_size: int,
) -> Path:
    """Write ``model.safetensors.index.json`` in the layout vLLM/HF read."""
    path = Path(out_dir) / SHARD_INDEX_NAME
    path.write_text(json.dumps(
        {
            "metadata": {"total_size": int(total_size)},
            "weight_map": dict(weight_map),
        },
        indent=2,
    ))
    return path


def describe_container_layout(names: Iterable[str]) -> tuple[str, int]:
    """Classify a published container set as ``(kind, count)``.

    ``kind`` is ``"single"`` for exactly ``model.safetensors``, ``"sharded"``
    for a complete ``model-XXXXX-of-YYYYY.safetensors`` run, and ``"other"``
    for anything a stock loader would not recognise as one model. Callers use
    it to decide whether an index file is required, forbidden, or unknown --
    the layout says which, so no caller has to be told.
    """
    observed = sorted(str(name) for name in names)
    if observed == [SINGLE_CONTAINER_NAME]:
        return "single", 1
    matches = [SHARD_NAME_RE.fullmatch(name) for name in observed]
    if matches and all(matches):
        counts = {int(m.group(2)) for m in matches if m is not None}
        indexes = sorted(int(m.group(1)) for m in matches if m is not None)
        if len(counts) == 1:
            count = counts.pop()
            if count == len(observed) and indexes == list(range(1, count + 1)):
                return "sharded", count
    return "other", len(observed)
