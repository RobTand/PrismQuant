#!/usr/bin/env python3
"""Merge unit-sharded ProductionWeightCache renders into one canonical cache.

Parallel producer v1 splits a render-bound cache build across boxes by unit
(``PRISMAQUANT_UNIT_SHARD=i/N``, see ``prismaquant/unit_sharding.py``). This
tool puts the pieces back together into the SAME ``ProductionWeightCache``
layout the unsharded stage writes — no parallel store, no new consumer.

The completeness gate is fail-closed and is the whole point of the tool. A
distributed render that quietly dropped a unit would produce a cost table
where that unit falls back to a different estimator, or an export that RTNs a
Linear by omission — both are silent-quality failures that no downstream gate
would name. So: every unit of the full enumeration must be accounted for
exactly once, by the shard the deterministic partition says owns it. Missing,
duplicated, or out-of-shard units are hard errors that name the units. There
is no ``--force``, no zero-fill, and no "merge what we have".

Usage:

    python tools/merge_unit_shards.py merge \\
        --shard /work/shard0/production_cache.pkl \\
        --shard /work/shard1/production_cache.pkl \\
        --output /work/artifacts/production_cache.pkl \\
        --output-cache-dir /work/artifacts/production_cache_dir

    python tools/merge_unit_shards.py digest \\
        --cache /work/artifacts/production_cache.pkl \\
        --output /work/artifacts/production_cache.digest.json

``digest`` writes the content manifest used to byte-compare a merged cache
against the unsharded reference: per ``(qname, format)`` SHA-256 over the
rendered tensor's exact bytes plus dtype/shape, the render-score records, and
``activation_max_abs``. It deliberately does NOT hash the pickle container —
a merged cache carries provenance an unsharded one cannot (which boxes
rendered what), and container bytes are not a stable equality test anyway.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from prismaquant import format_registry as fr  # noqa: E402
from prismaquant.unit_sharding import (  # noqa: E402
    SCHEMA as SHARD_SCHEMA,
    owed_pairs_from_stamp,
    partition_from_stamp,
)

MERGE_SCHEMA = "prismaquant.unit_shard_merge.v1"
DIGEST_SCHEMA = "prismaquant.production_cache_digest.v1"


def _load_cache(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _metadata(cache) -> Mapping:
    meta = getattr(cache, "metadata", None)
    return meta if isinstance(meta, Mapping) else {}


def _shard_stamp(cache, path: Path) -> Mapping:
    stamp = _metadata(cache).get("unit_shard")
    if not isinstance(stamp, Mapping):
        raise SystemExit(
            f"[merge-unit-shards] ERROR: {path} carries no 'unit_shard' "
            "stamp. It was not produced with PRISMAQUANT_UNIT_SHARD set, so "
            "there is nothing to merge — use it directly."
        )
    if str(stamp.get("schema")) != SHARD_SCHEMA:
        raise SystemExit(
            f"[merge-unit-shards] ERROR: {path} has unit-shard schema "
            f"{stamp.get('schema')!r}, expected {SHARD_SCHEMA!r}."
        )
    return stamp


def _resolve_weight_bytes(cache, key, base_dir: Path | None):
    """Return (tensor, on_disk_filename_or_None) for one cache entry."""
    value = cache.weights[key]
    if isinstance(value, str):
        if base_dir is None:
            raise SystemExit(
                f"[merge-unit-shards] ERROR: entry {key} is a path reference "
                f"({value!r}) but its cache has no cache_dir."
            )
        path = base_dir / value
        if not path.is_file():
            raise SystemExit(
                f"[merge-unit-shards] ERROR: entry {key} references missing "
                f"shard file {path}"
            )
        return None, value
    return value, None


def _tensor_digest(tensor: torch.Tensor) -> dict:
    contiguous = tensor.detach().cpu().contiguous()
    # Reinterpret as bytes: numpy has no bfloat16, and the comparison this
    # feeds is about exact rendered bytes, not about a float view of them.
    payload = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _entry_digest(cache, key, base_dir: Path | None) -> dict:
    value = cache.weights[key]
    if isinstance(value, str):
        tensor = torch.load(
            base_dir / value, map_location="cpu", weights_only=True
        )
    else:
        tensor = value
    return _tensor_digest(tensor)


def _canonical_unit_order(stamps) -> list[str]:
    partition = partition_from_stamp(stamps[0])
    return [name for name, _ in partition.units]


def _gate_shards(stamps, paths) -> None:
    """Fail closed unless the shards are exactly one complete partition."""
    hashes = {str(stamp["partition_hash"]) for stamp in stamps}
    if len(hashes) != 1:
        detail = ", ".join(
            f"{path.name}={stamp['partition_hash'][:16]}"
            for stamp, path in zip(stamps, paths)
        )
        raise SystemExit(
            "[merge-unit-shards] ERROR: shards do not share one partition; "
            f"partition_hash differs ({detail}). They split different unit "
            "enumerations and cannot be merged."
        )
    counts = {int(stamp["shard_count"]) for stamp in stamps}
    if len(counts) != 1:
        raise SystemExit(
            f"[merge-unit-shards] ERROR: shards disagree on shard_count: "
            f"{sorted(counts)}"
        )
    count = counts.pop()
    if len(stamps) != count:
        raise SystemExit(
            f"[merge-unit-shards] ERROR: partition declares {count} shards "
            f"but {len(stamps)} were supplied."
        )
    indices = [int(stamp["shard_index"]) for stamp in stamps]
    if sorted(indices) != list(range(count)):
        raise SystemExit(
            f"[merge-unit-shards] ERROR: shard indices {sorted(indices)} are "
            f"not exactly 0..{count - 1}; missing or duplicated shards."
        )

    partition = partition_from_stamp(stamps[0])
    for stamp in stamps:
        index = int(stamp["shard_index"])
        expected = list(partition.shards[index])
        observed = [str(name) for name in stamp["unit_names"]]
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            raise SystemExit(
                f"[merge-unit-shards] ERROR: shard {stamp['shard']} unit list "
                "does not match the recomputed partition slice; "
                f"missing={missing[:8]} extra={extra[:8]}"
            )


def _expected_pairs(cache, stamp, assignment) -> set[tuple[str, str]]:
    """The (qname, format) entries this shard was supposed to produce.

    A shard produced by a current build states its own debt in the stamp
    (``owed_pairs``), recorded from the same format map its render loop
    consumed. That is authoritative and keeps the operator's layer config out
    of the trust path. The reconstruction below is the fallback for stamps
    that predate the field.
    """
    stamped = owed_pairs_from_stamp(stamp)
    if stamped is not None:
        return stamped
    owned = [str(name) for name in stamp["unit_names"]]
    meta = _metadata(cache)
    scope = str(meta.get("render_scope", ""))
    if scope == "assignment":
        if assignment is None:
            raise SystemExit(
                "[merge-unit-shards] ERROR: assignment-scope shards require "
                "--render-layer-config so the merge knows which (unit, "
                "format) entries were owed."
            )
        pairs = set()
        for qname in owned:
            fmt = assignment.get(qname)
            if fmt is None:
                continue
            fmt_canon = fr.canonical_format_name(str(fmt))
            if fmt_canon == "BF16":
                continue
            pairs.add((qname, fmt_canon))
        return pairs
    formats = [
        fr.canonical_format_name(str(fmt))
        for fmt in (meta.get("requested_formats") or [])
    ]
    formats = [fmt for fmt in formats if fmt != "BF16"]
    if not formats:
        raise SystemExit(
            "[merge-unit-shards] ERROR: shard "
            f"{stamp['shard']} records no requested_formats; cannot decide "
            "what it owed."
        )
    return {(qname, fmt) for qname in owned for fmt in formats}


def _merge(args) -> int:
    paths = [Path(p) for p in args.shard]
    if len(paths) < 2:
        raise SystemExit(
            "[merge-unit-shards] ERROR: --shard must be given at least twice."
        )
    caches = [_load_cache(path) for path in paths]
    stamps = [_shard_stamp(cache, path) for cache, path in zip(caches, paths)]
    _gate_shards(stamps, paths)

    assignment = None
    if args.render_layer_config:
        from prismaquant.layer_config import load_assignment

        assignment = load_assignment(args.render_layer_config)

    order = {name: i for i, name in enumerate(_canonical_unit_order(stamps))}
    owner_by_unit: dict[str, int] = {}
    for stamp in stamps:
        for name in stamp["unit_names"]:
            owner_by_unit[str(name)] = int(stamp["shard_index"])

    merged_weights: dict[tuple[str, str], object] = {}
    seen_from: dict[tuple[str, str], str] = {}
    duplicates: list[str] = []
    out_of_shard: list[str] = []
    missing: list[str] = []
    failed: dict[tuple[str, str], str] = {}
    render_scores: dict[str, dict] = {}
    gate_records: list[dict] = []
    activation_max_abs: dict[str, float] = {}
    max_abs_conflicts: list[str] = []

    output_dir = Path(args.output_cache_dir) if args.output_cache_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for cache, stamp, path in zip(caches, stamps, paths):
        label = str(stamp["shard"])
        owned = set(str(name) for name in stamp["unit_names"])
        base_dir = Path(cache.cache_dir) if cache.cache_dir else path.parent
        expected = _expected_pairs(cache, stamp, assignment)
        observed = set(cache.weights)
        for key in sorted(observed, key=lambda k: (order.get(k[0], 1 << 30), k)):
            qname, fmt = str(key[0]), str(key[1])
            if qname not in owned:
                out_of_shard.append(f"{qname}@{fmt} in shard {label}")
                continue
            if key in seen_from:
                duplicates.append(
                    f"{qname}@{fmt} in shards {seen_from[key]} and {label}"
                )
                continue
            seen_from[key] = label
            tensor, filename = _resolve_weight_bytes(cache, key, base_dir)
            if filename is not None:
                if output_dir is None:
                    raise SystemExit(
                        "[merge-unit-shards] ERROR: shards are disk-streamed "
                        "(cache_dir set); --output-cache-dir is required."
                    )
                destination = output_dir / filename
                source = base_dir / filename
                if destination.resolve() != source.resolve():
                    if destination.exists():
                        destination.unlink()
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copy2(source, destination)
                merged_weights[key] = filename
            else:
                merged_weights[key] = tensor
        for owed in sorted(expected - observed):
            missing.append(f"{owed[0]}@{owed[1]} owed by shard {label}")
        for key, error in (cache.failed or {}).items():
            failed[(str(key[0]), str(key[1]))] = str(error)
        records = _metadata(cache).get("render_scores")
        if isinstance(records, Mapping):
            for record_key, record in (records.get("records") or {}).items():
                if record_key in render_scores:
                    duplicates.append(f"render score {record_key} in {label}")
                    continue
                render_scores[str(record_key)] = dict(record)
        gates = _metadata(cache).get("render_gates")
        if isinstance(gates, Mapping):
            for record in gates.get("records") or []:
                if isinstance(record, Mapping):
                    gate_records.append(dict(record))
        for qname, value in (cache.activation_max_abs or {}).items():
            previous = activation_max_abs.get(str(qname))
            if previous is not None and float(previous) != float(value):
                max_abs_conflicts.append(
                    f"{qname}: {previous!r} vs {value!r}"
                )
            activation_max_abs[str(qname)] = float(value)

    problems: list[str] = []
    if missing:
        problems.append(
            f"{len(missing)} missing entries: {missing[:8]}"
        )
    if duplicates:
        problems.append(
            f"{len(duplicates)} duplicate entries: {duplicates[:8]}"
        )
    if out_of_shard:
        problems.append(
            f"{len(out_of_shard)} entries outside their shard: "
            f"{out_of_shard[:8]}"
        )
    if failed:
        problems.append(
            f"{len(failed)} failed renders: {sorted(failed)[:8]}"
        )
    if max_abs_conflicts:
        # Every shard computes activation_max_abs over the FULL enumeration,
        # so disagreement means the shards did not see the same calibration
        # forward — the render bytes cannot be assumed comparable either.
        problems.append(
            f"{len(max_abs_conflicts)} activation_max_abs disagreements "
            f"across shards: {max_abs_conflicts[:8]}"
        )
    if problems:
        raise SystemExit(
            "[merge-unit-shards] ERROR: completeness gate failed.\n  "
            + "\n  ".join(problems)
        )

    # Canonical insertion order: full-enumeration unit order, then the
    # requested-format order — which is exactly the order an unsharded run
    # inserts its keys in. A merge that appended shard-by-shard, or that
    # sorted formats alphabetically, would produce a different pickle for the
    # same content and defeat any container-level comparison downstream.
    format_rank = {
        fr.canonical_format_name(str(fmt)): i
        for i, fmt in enumerate(_metadata(caches[0]).get("requested_formats") or [])
    }
    ordered_weights = {
        key: merged_weights[key]
        for key in sorted(
            merged_weights,
            key=lambda k: (
                order.get(k[0], 1 << 30),
                format_rank.get(k[1], 1 << 30),
                k[1],
            ),
        )
    }

    from prismaquant.production_weight_cache import (
        ProductionWeightCache,
        _summarize_render_gate_records,
        _write_render_score_sidecar,
    )

    reference_meta = dict(_metadata(caches[0]))
    reference_meta.pop("unit_shard", None)
    # Re-derive the render-gate summary over EVERY shard's records. Inheriting
    # shard 0's counters would label one shard's tally as the whole run's.
    gate_records.sort(
        key=lambda rec: (
            order.get(str(rec.get("qname", "")), 1 << 30),
            format_rank.get(str(rec.get("format", "")), 1 << 30),
            str(rec.get("format", "")),
        )
    )
    gate_summary = _summarize_render_gate_records(gate_records)
    reference_meta["render_gates"] = {**gate_summary, "records": gate_records}
    mechanisms = gate_summary.get("mechanisms") or {}
    reference_meta["four_over_six"] = dict(
        mechanisms.get("four_over_six")
        or {"accepted": 0, "rejected": 0, "package_accepted": 0, "reasons": {}}
    )
    reference_meta["render_scores"] = {
        "schema": "prismaquant.production_render_scores.v1",
        "entries": int(len(render_scores)),
        "records": dict(sorted(render_scores.items())),
        "cost_semantics": (
            (_metadata(caches[0]).get("render_scores") or {}).get(
                "cost_semantics", ""
            )
        ),
    }
    reference_meta["requested_entries"] = int(len(ordered_weights))
    reference_meta["render_failures"] = {}
    reference_meta["unit_shard_merge"] = {
        "schema": MERGE_SCHEMA,
        "partition_hash": str(stamps[0]["partition_hash"]),
        "shard_count": int(stamps[0]["shard_count"]),
        "total_units": int(stamps[0]["total_units"]),
        "merged_entries": int(len(ordered_weights)),
        "shards": [
            {
                "shard": str(stamp["shard"]),
                "unit_count": int(stamp["unit_count"]),
                "shard_bytes": int(stamp["shard_bytes"]),
                "source": str(path),
                "host": dict(stamp.get("host") or {}),
                "calib_hash": _metadata(cache).get("calib_hash"),
            }
            for stamp, path, cache in sorted(
                zip(stamps, paths, caches),
                key=lambda item: int(item[0]["shard_index"]),
            )
        ],
    }

    merged = ProductionWeightCache(
        weights=ordered_weights,
        levers=dict(caches[0].levers),
        activation_max_abs=activation_max_abs or None,
        failed={},
        cache_dir=str(output_dir) if output_dir is not None else None,
        metadata=reference_meta,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(merged, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if output_dir is not None:
        # Write the sidecars through the SAME writer the unsharded stage uses.
        # A merged cache_dir has to be resumable and re-readable exactly like
        # an unsharded one; a bare records dict here is silently discarded by
        # `_load_render_score_sidecar`, which reads `payload["records"]`.
        _write_render_score_sidecar(
            output_dir / "render_scores.json", render_scores
        )
        (output_dir / "activation_max_abs.json").write_text(
            json.dumps(
                {
                    name: activation_max_abs[name]
                    for name in sorted(
                        activation_max_abs, key=lambda n: order.get(n, 1 << 30)
                    )
                },
                indent=2,
            )
        )
    print(
        f"[merge-unit-shards] merged {len(stamps)} shards -> {output_path} "
        f"({len(ordered_weights)} entries, {len(render_scores)} render "
        f"scores, partition_hash={str(stamps[0]['partition_hash'])[:16]})",
        flush=True,
    )
    if args.digest_json:
        _write_digest(merged, Path(args.digest_json))
    return 0


def _write_digest(cache, output_path: Path) -> dict:
    base_dir = Path(cache.cache_dir) if cache.cache_dir else None
    entries = {
        f"{qname}|{fmt}": _entry_digest(cache, (qname, fmt), base_dir)
        for (qname, fmt) in sorted(cache.weights)
    }
    records = _metadata(cache).get("render_scores")
    scores = dict(sorted((records or {}).get("records", {}).items()))
    payload = {
        "schema": DIGEST_SCHEMA,
        "entries": entries,
        "entry_count": len(entries),
        "render_scores": scores,
        "activation_max_abs": dict(
            sorted((cache.activation_max_abs or {}).items())
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    print(
        f"[merge-unit-shards] digest {output_path} entries={len(entries)} "
        f"manifest_sha256={digest}",
        flush=True,
    )
    return payload


def _digest(args) -> int:
    cache = _load_cache(Path(args.cache))
    _write_digest(cache, Path(args.output))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    merge = sub.add_parser(
        "merge", help="Merge N unit-shard caches into one canonical cache."
    )
    merge.add_argument("--shard", action="append", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--output-cache-dir", default=None)
    merge.add_argument(
        "--render-layer-config",
        default=None,
        help="Required for assignment-scope shards: the concrete assignment "
        "that says which (unit, format) entries each shard owed.",
    )
    merge.add_argument("--digest-json", default=None)
    merge.set_defaults(func=_merge)

    digest = sub.add_parser(
        "digest", help="Write a content digest manifest for one cache."
    )
    digest.add_argument("--cache", required=True)
    digest.add_argument("--output", required=True)
    digest.set_defaults(func=_digest)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
