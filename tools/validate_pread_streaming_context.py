#!/usr/bin/env python3
"""Exercise pread through the real streaming context and resident cache.

This bounded GPU validation builds the model's normal streaming skeleton,
prefetches one requested layer through the explicit pread source backend,
requires that prefetch on install, unloads the layer, and proves that a second
required access is served by the existing ``LayerCache``.  It does not run a
model forward and therefore makes no throughput or numeric-quality claim.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch

from prismaquant.streaming_model import _build_streaming_context


SCHEMA = "prismaquant.pread-streaming-context-validation.v1"


class ValidationError(RuntimeError):
    pass


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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or tmp.exists():
        raise ValidationError(f"refusing to overwrite output: {path}")
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tmp.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offload-folder", required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--cache-headroom-gb", type=float, default=80.0)
    parser.add_argument("--prefetch-min-available-gb", type=float, default=60.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.layer < 0:
        raise ValidationError("--layer must be non-negative")
    if not torch.cuda.is_available():
        raise ValidationError("CUDA is required for streaming-context validation")
    model = Path(args.model).resolve()
    output = Path(args.output).resolve()
    offload = Path(args.offload_folder).resolve()
    if not model.is_dir():
        raise ValidationError(f"model directory is absent: {model}")
    if output.exists():
        raise ValidationError(f"refusing to overwrite output: {output}")
    offload.mkdir(parents=True, exist_ok=True)

    before_io = _proc_io()
    before_usage = resource.getrusage(resource.RUSAGE_SELF)
    before_fds = _fd_targets()
    torch.cuda.reset_peak_memory_stats()
    start_ns = time.monotonic_ns()
    ctx = None
    first_source = None
    second_source = None
    layer_tensor_count = 0
    layer_payload_bytes = 0
    cache_before: dict[str, Any] = {}
    cache_after: dict[str, Any] = {}
    try:
        ctx = _build_streaming_context(
            str(model),
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            offload_folder=str(offload),
            cache_headroom_gb=args.cache_headroom_gb,
            max_cache_slots=2,
            prefetch_workers=1,
            prefetch_min_available_gb=args.prefetch_min_available_gb,
            log_prefix="[pread-context-validation]",
            safetensors_backend="pread",
        )
        if ctx.safetensors_backend != "pread":
            raise ValidationError(
                f"streaming context resolved backend {ctx.safetensors_backend!r}"
            )
        if args.layer >= ctx.num_layers:
            raise ValidationError(
                f"requested layer {args.layer}, model has {ctx.num_layers} layers"
            )
        future = ctx.schedule_prefetch(args.layer)
        if future is None:
            raise ValidationError("streaming context refused the bounded prefetch")
        first_source = ctx.install(args.layer, require_prefetched=True)
        live_layer = ctx.layers[args.layer]
        meta_after_install = [
            name for name, value in (
                *live_layer.named_parameters(recurse=True),
                *live_layer.named_buffers(recurse=True),
            )
            if getattr(value, "is_meta", False)
        ]
        if meta_after_install:
            raise ValidationError(
                f"installed layer retains meta tensors: {meta_after_install[:8]}"
            )
        if not ctx.layer_cache.peek(args.layer):
            raise ValidationError("prefetched layer was not retained in LayerCache")
        cached = ctx.layer_cache._cache[args.layer]
        layer_tensor_count = len(cached)
        layer_payload_bytes = sum(
            int(t.numel() * t.element_size()) for t in cached.values()
        )
        cache_before = {
            "entries": len(ctx.layer_cache._cache),
            "current_bytes": ctx.layer_cache.total_bytes,
            "summary": str(ctx.layer_cache),
        }
        ctx.unload(args.layer)
        _tensors, second_source = ctx.ensure_loaded(
            args.layer, require_prefetched=True
        )
        if second_source != "hot":
            raise ValidationError(
                f"second required access was {second_source!r}, expected 'hot'"
            )
        cache_after = {
            "entries": len(ctx.layer_cache._cache),
            "current_bytes": ctx.layer_cache.total_bytes,
            "summary": str(ctx.layer_cache),
        }
    finally:
        if ctx is not None:
            try:
                ctx.unload(args.layer)
            except Exception:
                pass
            ctx.shutdown()
        del ctx
        gc.collect()
        torch.cuda.empty_cache()

    end_ns = time.monotonic_ns()
    after_io = _proc_io()
    after_usage = resource.getrusage(resource.RUSAGE_SELF)
    after_fds = _fd_targets()
    leaked = {
        fd: target for fd, target in after_fds.items()
        if fd not in before_fds and str(model) in target
    }
    if leaked:
        raise ValidationError(f"checkpoint descriptors leaked: {leaked}")

    _atomic_json(output, {
        "schema": SCHEMA,
        "status": "PASS",
        "scope": "bounded_real_model_streaming_context",
        "production_claim": False,
        "throughput_claim": False,
        "model": str(model),
        "layer": args.layer,
        "backend": "pread",
        "first_required_access_source": first_source,
        "second_required_access_source": second_source,
        "layer_tensor_count": layer_tensor_count,
        "layer_payload_bytes": layer_payload_bytes,
        "cache_before_unload": cache_before,
        "cache_after_required_reuse": cache_after,
        "wall_ns": end_ns - start_ns,
        "proc_io_delta": {
            key: after_io.get(key, 0) - before_io.get(key, 0)
            for key in after_io
        },
        "rusage_delta": {
            "minor_faults": after_usage.ru_minflt - before_usage.ru_minflt,
            "major_faults": after_usage.ru_majflt - before_usage.ru_majflt,
            "in_blocks": after_usage.ru_inblock - before_usage.ru_inblock,
            "out_blocks": after_usage.ru_oublock - before_usage.ru_oublock,
            "max_rss_kib": after_usage.ru_maxrss,
        },
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "fd_count_before": len(before_fds),
        "fd_count_after": len(after_fds),
        "checkpoint_fd_leaks": leaked,
        "python": sys.version,
        "torch": torch.__version__,
    })
    print(output)


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
