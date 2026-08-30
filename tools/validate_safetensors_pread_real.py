#!/usr/bin/env python3
"""Bounded real-checkpoint validation for the streamed pread backend.

The public mode runs ``safe_open`` and ``pread`` in fresh child processes,
then proves that both readers returned the same names, shapes, dtypes, byte
counts, and raw tensor SHA-256 digests.  Per-process ``/proc/self/io`` and
``getrusage`` deltas are evidence about the read paths; they are deliberately
recorded without making a throughput claim because this tool neither drops nor
controls the host page cache.

This is an offline validation tool.  Production layer loads continue through
``layer_streaming._read_layer_to_device`` and the existing resident prefetch
and ``LayerCache`` machinery.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from prismaquant.safetensors_pread import PreadSafetensors


SCHEMA = "prismaquant.safetensors-pread-real-validation.v1"
WORKER_SCHEMA = "prismaquant.safetensors-pread-real-worker.v1"


class ValidationError(RuntimeError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(), object_pairs_hook=_strict_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read strict JSON {path}: {exc}") from exc


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or tmp.exists():
        raise ValidationError(f"refusing to overwrite validation output: {path}")
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tmp.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proc_io() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, raw = line.split(":", 1)
        result[key] = int(raw.strip())
    return result


def _fd_targets() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in Path("/proc/self/fd").iterdir():
        try:
            result[path.name] = os.readlink(path)
        except FileNotFoundError:
            pass
    return result


def _delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in after}


def _tensor_digest(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    byte_view = contiguous.view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(byte_view)).hexdigest()


def _validated_selection(model: Path, selection_path: Path) -> list[dict[str, str]]:
    raw = _load_json(selection_path)
    if not isinstance(raw, list) or not raw:
        raise ValidationError("selection must be a non-empty JSON array of tensor names")
    if any(not isinstance(name, str) or not name for name in raw):
        raise ValidationError("every selection entry must be a non-empty string")
    if len(set(raw)) != len(raw):
        raise ValidationError("selection contains duplicate tensor names")
    index_path = model / "model.safetensors.index.json"
    index = _load_json(index_path)
    try:
        weight_map = index["weight_map"]
    except (KeyError, TypeError) as exc:
        raise ValidationError(f"invalid safetensors index {index_path}") from exc
    result: list[dict[str, str]] = []
    for name in raw:
        shard = weight_map.get(name)
        if not isinstance(shard, str) or not shard:
            raise ValidationError(f"selected tensor is absent from index: {name}")
        shard_path = model / shard
        if not shard_path.is_file():
            raise ValidationError(f"selected shard is absent: {shard_path}")
        result.append({"name": name, "shard": shard})
    return result


def _run_worker(args: argparse.Namespace) -> None:
    model = Path(args.model).resolve()
    selection_path = Path(args.selection).resolve()
    output = Path(args.worker_output).resolve()
    selection = _validated_selection(model, selection_path)
    backend = args.worker_backend
    if backend not in {"safe_open", "pread"}:
        raise ValidationError(f"invalid worker backend: {backend!r}")

    before_io = _proc_io()
    before_usage = resource.getrusage(resource.RUSAGE_SELF)
    before_fds = _fd_targets()
    wall_start = time.monotonic_ns()
    process_start = time.process_time_ns()
    rows: list[dict[str, Any]] = []
    shard_identities: dict[str, dict[str, int]] = {}

    by_shard: dict[str, list[str]] = {}
    for item in selection:
        by_shard.setdefault(item["shard"], []).append(item["name"])
    for shard, names in by_shard.items():
        shard_path = model / shard
        st = shard_path.stat()
        shard_identities[shard] = {
            "device": int(st.st_dev),
            "inode": int(st.st_ino),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "ctime_ns": int(st.st_ctime_ns),
        }
        reader = (
            safe_open(str(shard_path), framework="pt")
            if backend == "safe_open"
            else PreadSafetensors(shard_path)
        )
        with reader as handle:
            available = set(handle.keys())
            for name in names:
                if name not in available:
                    raise ValidationError(f"{name!r} absent from indexed shard {shard}")
                tensor = handle.get_tensor(name)
                rows.append({
                    "name": name,
                    "shard": shard,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "nbytes": int(tensor.numel() * tensor.element_size()),
                    "sha256": _tensor_digest(tensor),
                })
                del tensor
        del reader
        gc.collect()

    process_end = time.process_time_ns()
    wall_end = time.monotonic_ns()
    after_fds = _fd_targets()
    after_usage = resource.getrusage(resource.RUSAGE_SELF)
    after_io = _proc_io()
    leaked_targets = {
        fd: target for fd, target in after_fds.items()
        if fd not in before_fds and str(model) in target
    }
    if leaked_targets:
        raise ValidationError(f"checkpoint descriptors leaked: {leaked_targets}")

    _atomic_json(output, {
        "schema": WORKER_SCHEMA,
        "status": "ok",
        "backend": backend,
        "model": str(model),
        "selection_path": str(selection_path),
        "selection_sha256": _sha256_file(selection_path),
        "pid": os.getpid(),
        "python": sys.version,
        "torch": torch.__version__,
        "cache_control": "none",
        "timing_scope": "selected_tensor_materialization_and_sha256",
        "wall_ns": wall_end - wall_start,
        "process_cpu_ns": process_end - process_start,
        "proc_io_delta": _delta(after_io, before_io),
        "rusage_delta": {
            "minor_faults": after_usage.ru_minflt - before_usage.ru_minflt,
            "major_faults": after_usage.ru_majflt - before_usage.ru_majflt,
            "in_blocks": after_usage.ru_inblock - before_usage.ru_inblock,
            "out_blocks": after_usage.ru_oublock - before_usage.ru_oublock,
            "voluntary_context_switches": after_usage.ru_nvcsw - before_usage.ru_nvcsw,
            "involuntary_context_switches": after_usage.ru_nivcsw - before_usage.ru_nivcsw,
            "max_rss_kib": after_usage.ru_maxrss,
        },
        "fd_count_before": len(before_fds),
        "fd_count_after": len(after_fds),
        "checkpoint_fd_leaks": leaked_targets,
        "shard_identity": shard_identities,
        "tensors": rows,
    })


def _compare(args: argparse.Namespace) -> None:
    model = Path(args.model).resolve()
    selection = Path(args.selection).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ValidationError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    worker_receipts: dict[str, dict[str, Any]] = {}
    command_rows: dict[str, list[str]] = {}
    for backend in ("safe_open", "pread"):
        receipt = output_dir / f"{backend}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model", str(model),
            "--selection", str(selection),
            "--worker-backend", backend,
            "--worker-output", str(receipt),
        ]
        command_rows[backend] = command
        subprocess.run(command, check=True)
        row = _load_json(receipt)
        if row.get("schema") != WORKER_SCHEMA or row.get("status") != "ok":
            raise ValidationError(f"invalid worker receipt: {receipt}")
        worker_receipts[backend] = row

    baseline = worker_receipts["safe_open"]
    candidate = worker_receipts["pread"]
    fields = ("selection_sha256", "shard_identity", "tensors")
    mismatches = [field for field in fields if baseline[field] != candidate[field]]
    if mismatches:
        raise ValidationError(f"backend mismatch in fields: {', '.join(mismatches)}")
    receipt_path = output_dir / "receipt.json"
    _atomic_json(receipt_path, {
        "schema": SCHEMA,
        "status": "PASS",
        "scope": "offline_real_checkpoint_source_reader_validation",
        "production_claim": False,
        "throughput_claim": False,
        "cache_control": "none",
        "note": (
            "Timings and process counters are observations only; host page cache "
            "was not controlled. This receipt proves exact selected-tensor parity "
            "and descriptor cleanup, not production throughput."
        ),
        "model": str(model),
        "selection_path": str(selection),
        "selection_sha256": baseline["selection_sha256"],
        "selected_tensor_count": len(baseline["tensors"]),
        "selected_shard_count": len(baseline["shard_identity"]),
        "selected_payload_bytes": sum(row["nbytes"] for row in baseline["tensors"]),
        "matched_fields": list(fields),
        "commands": command_rows,
        "workers": {
            backend: {
                "receipt": str(output_dir / f"{backend}.json"),
                "wall_ns": worker_receipts[backend]["wall_ns"],
                "process_cpu_ns": worker_receipts[backend]["process_cpu_ns"],
                "proc_io_delta": worker_receipts[backend]["proc_io_delta"],
                "rusage_delta": worker_receipts[backend]["rusage_delta"],
                "fd_count_before": worker_receipts[backend]["fd_count_before"],
                "fd_count_after": worker_receipts[backend]["fd_count_after"],
                "checkpoint_fd_leaks": worker_receipts[backend]["checkpoint_fd_leaks"],
            }
            for backend in ("safe_open", "pread")
        },
    })
    print(receipt_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--worker-backend", choices=("safe_open", "pread"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.worker_backend or args.worker_output:
        if not args.worker_backend or not args.worker_output or args.output_dir:
            raise ValidationError("worker mode requires both hidden worker arguments only")
        _run_worker(args)
    else:
        if not args.output_dir:
            raise ValidationError("--output-dir is required")
        _compare(args)


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
