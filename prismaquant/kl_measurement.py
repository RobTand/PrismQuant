"""Production KL measurement utilities.

This is the live home for whole-assignment KL, lane-batched per-candidate KL,
and the bounded CUDA graph helpers used by validation.  The implementation
preserves the measured production paths so archival of the cross-layer
entrypoints does not change KL semantics.
"""
from __future__ import annotations

import gc
import math
import os
import re
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import cost_entry_predicted_dloss
from prismaquant.allocator_solver import _shape_from_stats
from prismaquant.build_rtn_cache import kl_divergence
from prismaquant.memory_management import (
    GPUMemoryBudgetExceeded,
    cuda_memory_info,
    enforce_gpu_memory_budget,
    env_flag_enabled as _env_flag_enabled,
    env_float as _env_float,
    env_int as _env_int,
    max_gpu_memory_bytes,
    register_budget_evictor,
)
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    calibration_data_hash,
)
from prismaquant.nvfp4_activation_contract import ActivationScaleContractError
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    is_cb_format,
    validate_cb_assignment_serialization_stamps,
)
from prismaquant.footprint import (
    NVFP4_WEIGHT_ONLY_STATS_KEY,
    nvfp4_global_sidecar_bytes,
)

KLScope = Literal["last_token", "full_sequence"]


def resolve_kl_scope(kl_scope: KLScope | None = None) -> KLScope:
    """Resolve KL reduction scope, preserving the legacy env override.

    Passing an explicit ``kl_scope`` wins.  ``None`` means use
    ``PRISMAQUANT_FULL_SEQUENCE_KL`` for backward compatibility, with
    last-token KL as the default when the env var is unset.
    """
    if kl_scope is not None:
        if kl_scope not in {"last_token", "full_sequence"}:
            raise ValueError(
                "kl_scope must be 'last_token' or 'full_sequence', "
                f"got {kl_scope!r}"
            )
        return kl_scope
    return (
        "full_sequence"
        if _env_flag_enabled("PRISMAQUANT_FULL_SEQUENCE_KL", default=False)
        else "last_token"
    )


def _env_cuda_graphs_enabled_for_call_count(
    name: str,
    *,
    default: str | bool = "auto",
    call_count: int,
    min_calls: int,
) -> bool:
    """Return whether a CUDA graph path should run for this call pattern.

    L3 and coord-descent candidates are often one-shot graph keys. Capturing
    those graphs costs warmup + capture work without enough replays to pay it
    back, so the default is ``auto``: graph only when the same key is expected
    to run at least ``min_calls`` times. Explicit env values keep their force
    semantics: ``1``/``true`` force on, ``0``/``false`` force off.
    """

    value = os.environ.get(name)
    mode = default if value is None else value.strip().lower()
    if isinstance(mode, bool):
        return mode
    if mode in {"1", "true", "yes", "on", "force"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    if mode != "auto":
        return bool(default) if isinstance(default, bool) else False

    threshold = _env_int(f"{name}_MIN_CALLS", int(min_calls))
    return int(call_count) >= max(int(threshold), 1)


_PRISMAQUANT_GRAPH_POOL = None
_NOCLONE_OVERRIDE_WARNED = False


def get_prismaquant_graph_pool():
    """Return the process-wide CUDA graph memory pool, or None when disabled.

    Default ON: in-process tests confirm shared and private pools produce
    bit-identical captured-graph outputs (single graph and multi-registry).
    The earlier "5% NaN with shared pool" signal was process-init noise of
    the small Qwen-0.6B smoke -- 5 separate processes with identical config
    produce 5 different KLs whether the pool is shared or private. Set
    PRISMAQUANT_GRAPH_SHARED_POOL=0 only as a diagnostic.
    """
    if not _env_flag_enabled("PRISMAQUANT_GRAPH_SHARED_POOL", default=True):
        return None
    if not torch.cuda.is_available():
        return None
    global _PRISMAQUANT_GRAPH_POOL
    if _PRISMAQUANT_GRAPH_POOL is None:
        _PRISMAQUANT_GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _PRISMAQUANT_GRAPH_POOL


def _cuda_graph_pool_id(pool) -> str:
    if pool is None:
        return "private"
    return f"shared:{id(pool):x}"


def get_prismaquant_graph_pool_id() -> str:
    if not _env_flag_enabled("PRISMAQUANT_GRAPH_SHARED_POOL", default=True):
        return "private"
    if not torch.cuda.is_available():
        return "unavailable"
    if _PRISMAQUANT_GRAPH_POOL is None:
        return "shared:uninitialized"
    return _cuda_graph_pool_id(_PRISMAQUANT_GRAPH_POOL)


def _cost_entry(costs: Mapping, name: str, fmt: str) -> dict | None:
    per_name = costs.get(name, {})
    if not isinstance(per_name, Mapping):
        return None
    for alias in fr.aliases_for(fmt):
        entry = per_name.get(alias)
        if isinstance(entry, dict) and "error" not in entry:
            return entry
    return None


def l2_cost_value(stats: Mapping, costs: Mapping, name: str, fmt: str) -> float | None:
    """Return the allocator's L2 scalar cost for one existing cost entry."""
    entry = _cost_entry(costs, name, fmt)
    if entry is None or name not in stats:
        return None
    return float(
        cost_entry_predicted_dloss(stats[name], entry, format_name=fmt)
    )


def _memory_bytes_for_format(
    stats_entry: Mapping,
    spec: fr.FormatSpec,
) -> int:
    memory_map = stats_entry.get("_memory_bytes_by_format")
    if isinstance(memory_map, Mapping):
        # Legacy stats may be keyed by the pre-canonical name (e.g.
        # ``"MXFP8"`` before the alias to ``"MXFP8_E4M3"`` was added).
        # Try the canonical name first, then any registered aliases for
        # the same spec.
        for key in (spec.name, *fr.aliases_for(spec.name)):
            if key in memory_map:
                return int(memory_map[key])
    return int(spec.memory_bytes_for_shape(_shape_from_stats(dict(stats_entry))))


def assignment_bit_total(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    specs_by_name: Mapping[str, fr.FormatSpec],
    *,
    cb_serialization_context: CBSerializationContext | None = None,
    cb_serialization_stamps: Mapping[str, object] | None = None,
    where: str = "assignment_bit_total",
) -> float:
    """Return exact assignment payload bits, including shared CB sidecars.

    Non-CB formats preserve the historical FormatSpec/stat-memory-map path,
    with NVFP4's emitted global scale tensors added explicitly.
    CB formats are priced as one assignment so FP8 row scales and each
    physical codebook sidecar are charged exactly once. Persisted per-layer
    identities are mandatory: a format label alone cannot establish whether
    the assignment describes FP4 layout-v1 or v2, nor its sidecar sharing.
    """
    total = 0.0
    cb_assignment: dict[str, str] = {}
    cb_shapes: dict[str, tuple[int, ...]] = {}
    for name, fmt in assignment.items():
        if name not in stats:
            continue
        spec = specs_by_name[fr.canonical_format_name(fmt)]
        if is_cb_format(spec.name):
            cb_assignment[str(name)] = spec.name
            cb_shapes[str(name)] = _shape_from_stats(dict(stats[name]))
        else:
            total += 8.0 * _memory_bytes_for_format(stats[name], spec)
            if spec.name == "NVFP4":
                total += 8.0 * nvfp4_global_sidecar_bytes(
                    str(name),
                    _shape_from_stats(dict(stats[name])),
                    weight_only=bool(
                        stats[name].get(NVFP4_WEIGHT_ONLY_STATS_KEY, False)
                    ),
                )
    if cb_assignment:
        if cb_serialization_context is None:
            raise ValueError(
                f"{where}: CB assignment requires a CBSerializationContext"
            )
        validate_cb_assignment_serialization_stamps(
            cb_assignment,
            cb_shapes,
            context=cb_serialization_context,
            stamps=cb_serialization_stamps,
            where=where,
        )
        payload = cb_assignment_payload_breakdown(
            cb_assignment,
            cb_shapes,
            context=cb_serialization_context,
        )
        total += 8.0 * int(payload["total_bytes"])
    return total


def _tensor_tree_signature(value):
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__name__,
            tuple(
                sorted(
                    (str(key), _tensor_tree_signature(child))
                    for key, child in value.items()
                )
            ),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_tensor_tree_signature(child) for child in value))
    if isinstance(value, list):
        return ("list", tuple(_tensor_tree_signature(child) for child in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", type(value).__name__, value)
    return ("object", type(value).__name__, id(value))


def _clone_static_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_static_tree(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_static_tree(child) for child in value)
    if isinstance(value, list):
        return [_clone_static_tree(child) for child in value]
    return value


def _copy_static_tree(src, dst) -> bool:
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        if (
            tuple(src.shape) != tuple(dst.shape)
            or src.dtype != dst.dtype
            or src.device != dst.device
        ):
            return False
        dst.copy_(src)
        return True
    if isinstance(src, Mapping) and isinstance(dst, Mapping):
        if set(src.keys()) != set(dst.keys()):
            return False
        return all(_copy_static_tree(src[key], dst[key]) for key in src)
    if isinstance(src, tuple) and isinstance(dst, tuple):
        if len(src) != len(dst):
            return False
        return all(_copy_static_tree(a, b) for a, b in zip(src, dst))
    if isinstance(src, list) and isinstance(dst, list):
        if len(src) != len(dst):
            return False
        return all(_copy_static_tree(a, b) for a, b in zip(src, dst))
    if src is dst:
        return True
    if src is None or isinstance(src, (bool, int, float, str)):
        return src == dst
    return False


def _first_cuda_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value if value.is_cuda else None
    if isinstance(value, Mapping):
        for child in value.values():
            found = _first_cuda_tensor(child)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for child in value:
            found = _first_cuda_tensor(child)
            if found is not None:
                return found
    return None


def _clone_cuda_graph_output(value):
    clone_disabled = not _env_flag_enabled(
        "PRISMAQUANT_GRAPH_OUTPUT_CLONE",
        default=True,
    )
    if clone_disabled and _env_flag_enabled(
        "PRISMAQUANT_GRAPH_SHARED_POOL",
        default=True,
    ):
        global _NOCLONE_OVERRIDE_WARNED
        if not _NOCLONE_OVERRIDE_WARNED:
            _NOCLONE_OVERRIDE_WARNED = True
            print(
                "[cuda-graphs] warning: "
                "PRISMAQUANT_GRAPH_OUTPUT_CLONE=0 is unsafe with "
                "PRISMAQUANT_GRAPH_SHARED_POOL=1; cloning CUDA graph outputs instead",
                file=sys.stderr,
                flush=True,
            )
        clone_disabled = False
    if clone_disabled:
        return value
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_cuda_graph_output(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_cuda_graph_output(child) for child in value)
    if isinstance(value, list):
        return [_clone_cuda_graph_output(child) for child in value]
    return value


_CUDA_GRAPH_WARNED_LABELS: set[str] = set()


def _warn_cuda_graph_fallback_once(label: str, exc: BaseException) -> None:
    if label in _CUDA_GRAPH_WARNED_LABELS:
        return
    _CUDA_GRAPH_WARNED_LABELS.add(label)
    print(
        "[cuda-graphs] warning: "
        f"{label} capture/replay failed once; using eager for that shape "
        f"({type(exc).__name__}: {exc})",
        file=sys.stderr,
        flush=True,
    )


def _cuda_graph_debug_node_count(path: Path) -> int | None:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if "label=" in stripped and "->" not in stripped:
            count += 1
    return count if count > 0 else None


@dataclass
class _CUDAGraphEntry:
    graph: object
    static_args: tuple
    static_kwargs: dict
    static_output: object
    keepalive: tuple[object, ...] = field(default_factory=tuple)


class CUDAGraphRegistry:
    """Bounded LRU CUDA graph cache for fixed-shape tensor forwards.

    Each entry owns graph activation memory plus static input/output tensors.
    The default cap is intentionally small and can be overridden per path with
    the registry's ``max_entries_env`` variable.

    With ``PRISMAQUANT_GRAPH_SHARED_POOL`` enabled,
    ``PRISMAQUANT_GRAPH_OUTPUT_CLONE=0`` is ignored and outputs are cloned.
    Shared-pool captures can alias pool-mate static outputs still held by callers.
    """

    def __init__(
        self,
        *,
        label: str,
        max_entries: int = 4,
        max_entries_env: str | None = None,
        warmup_iters: int = 2,
        verbose_env: str | None = None,
    ):
        self.label = str(label)
        self.default_max_entries = max(int(max_entries), 0)
        self.max_entries_env = max_entries_env
        self.warmup_iters = max(int(warmup_iters), 0)
        self.verbose_env = verbose_env
        self.entries: OrderedDict[tuple, _CUDAGraphEntry] = OrderedDict()
        self.disabled_keys: set[tuple] = set()
        self.eviction_count = 0
        register_budget_evictor(self)

    def graph_pool(self):
        return get_prismaquant_graph_pool()

    def graph_pool_id(self) -> str:
        return get_prismaquant_graph_pool_id()

    def clear(self) -> None:
        had_entries = bool(self.entries)
        self.entries.clear()
        self.disabled_keys.clear()
        if had_entries and torch.cuda.is_available():
            gc.collect()
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _max_entries(self) -> int:
        if (
            self.max_entries_env is not None
            and os.environ.get(self.max_entries_env) is not None
        ):
            return _env_int(self.max_entries_env, self.default_max_entries)
        return _env_int(
            "PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH",
            self.default_max_entries,
        )

    def _verbose_enabled(self) -> bool:
        return (
            self.verbose_env is not None
            and _env_flag_enabled(self.verbose_env, default=False)
        )

    def _verbose_log(self, label: str, message: str) -> None:
        if not self._verbose_enabled():
            return
        print(
            f"[cuda-graphs][{self.label}:{label}] "
            f"{time.time():.6f} {message}",
            file=sys.stderr,
            flush=True,
        )

    def _verbose_exception(
        self,
        label: str,
        message: str,
        exc: BaseException,
    ) -> None:
        if not self._verbose_enabled():
            return
        self._verbose_log(label, f"{message}: {type(exc).__name__}: {exc}")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)

    def _debug_graph_summary(
        self,
        graph,
        label: str,
    ) -> tuple[int | str, str | None]:
        node_count: int | str = "unavailable"
        dump_path: str | None = None
        for attr in ("num_nodes", "_num_nodes"):
            fn = getattr(graph, attr, None)
            if callable(fn):
                try:
                    node_count = int(fn())
                    break
                except Exception:
                    pass
        debug_dump = getattr(graph, "debug_dump", None)
        if callable(debug_dump):
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{self.label}_{label}")
            path = (
                Path(tempfile.gettempdir())
                / f"prismaquant_cuda_graph_{safe}_{os.getpid()}.dot"
            )
            try:
                debug_dump(str(path))
                dump_path = str(path)
                if node_count == "unavailable":
                    parsed_count = _cuda_graph_debug_node_count(path)
                    if parsed_count is not None:
                        node_count = parsed_count
            except Exception as exc:
                self._verbose_exception(label, "debug dump failed", exc)
        return node_count, dump_path

    def _evict_if_needed(self) -> None:
        max_entries = self._max_entries()
        if max_entries <= 0:
            evicted = len(self.entries)
            self.entries.clear()
            self.eviction_count += evicted
            if evicted:
                self._log_graph_eviction(evicted, max_entries)
            return
        while len(self.entries) > max_entries:
            self._evict_oldest_graph_entry(max_entries=max_entries)

    def _log_graph_eviction(self, count: int, *, max_entries: int) -> None:
        print(
            "[cuda-graphs] "
            f"{self.label}: evicted {count} graph(s) "
            f"(max_entries={max_entries})",
            file=sys.stderr,
            flush=True,
        )

    def _evict_oldest_graph_entry(self, *, max_entries: int | None = None) -> bool:
        if not self.entries:
            return False
        self.entries.popitem(last=False)
        self.eviction_count += 1
        self._log_graph_eviction(
            1,
            max_entries=self._max_entries() if max_entries is None else max_entries,
        )
        return True

    def evict_oldest_for_memory_budget(self) -> bool:
        return self._evict_oldest_graph_entry()

    def _cleanup_failed_capture(
        self,
        graph,
        device: torch.device,
        label: str,
    ) -> None:
        if graph is not None:
            reset = getattr(graph, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception as exc:
                    self._verbose_exception(label, "failed graph reset failed", exc)
        try:
            torch.cuda.synchronize(device)
        except Exception as exc:
            self._verbose_exception(label, "post-failure synchronize failed", exc)
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            self._verbose_exception(label, "post-failure empty_cache failed", exc)

    def run(
        self,
        label: str,
        key: tuple,
        fn: Callable,
        *args,
        enabled: bool = True,
        device: torch.device | None = None,
        keepalive: tuple[object, ...] = (),
        **kwargs,
    ):
        cuda_tensor = _first_cuda_tensor((args, kwargs))
        graph_device = device
        if graph_device is None and cuda_tensor is not None:
            graph_device = cuda_tensor.device
        if (
            not enabled
            or not torch.cuda.is_available()
            or graph_device is None
            or torch.device(graph_device).type != "cuda"
            or self._max_entries() <= 0
        ):
            return fn(*args, **kwargs)

        full_key = (
            self.label,
            str(label),
            tuple(key),
            _tensor_tree_signature(args),
            _tensor_tree_signature(kwargs),
        )
        entry = self.entries.get(full_key)
        if entry is not None:
            self.entries.move_to_end(full_key)
            if not (
                _copy_static_tree(tuple(args), entry.static_args)
                and _copy_static_tree(dict(kwargs), entry.static_kwargs)
            ):
                return fn(*args, **kwargs)
            try:
                entry.graph.replay()
                return _clone_cuda_graph_output(entry.static_output)
            except Exception as exc:
                self.entries.pop(full_key, None)
                self.disabled_keys.add(full_key)
                self._verbose_exception(str(label), "replay failed", exc)
                _warn_cuda_graph_fallback_once(str(label), exc)
                return fn(*args, **kwargs)
        if full_key in self.disabled_keys:
            return fn(*args, **kwargs)

        try:
            enforce_gpu_memory_budget(
                [self],
                device=torch.device(graph_device),
                reason=f"{self.label}:{label} CUDA graph capture",
            )
            entry = self._capture(
                fn,
                args,
                kwargs,
                torch.device(graph_device),
                keepalive=keepalive,
                capture_label=str(label),
            )
        except Exception as exc:
            if isinstance(exc, GPUMemoryBudgetExceeded):
                raise
            self.disabled_keys.add(full_key)
            self._verbose_exception(str(label), "capture failed", exc)
            _warn_cuda_graph_fallback_once(str(label), exc)
            return fn(*args, **kwargs)
        self.entries[full_key] = entry
        self._evict_if_needed()
        enforce_gpu_memory_budget(
            [self],
            device=torch.device(graph_device),
            reason=f"{self.label}:{label} CUDA graph capture",
        )
        return _clone_cuda_graph_output(entry.static_output)

    def _capture(
        self,
        fn: Callable,
        args: tuple,
        kwargs: Mapping,
        device: torch.device,
        *,
        keepalive: tuple[object, ...],
        capture_label: str | None = None,
    ) -> _CUDAGraphEntry:
        label = capture_label or self.label
        capture_start_wall = time.time()
        capture_start = time.perf_counter()
        self._verbose_log(
            label,
            f"capture start device={device} warmup_iters={self.warmup_iters}",
        )
        static_args = tuple(_clone_static_tree(value) for value in args)
        static_kwargs = {
            key: _clone_static_tree(value)
            for key, value in dict(kwargs).items()
        }
        current_stream = torch.cuda.current_stream(device)
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream), torch.no_grad():
            for _ in range(self.warmup_iters):
                fn(*static_args, **static_kwargs)
        current_stream.wait_stream(side_stream)
        if self._verbose_enabled():
            try:
                torch.cuda.synchronize(device)
            except Exception as exc:
                self._verbose_exception(label, "warmup synchronize failed", exc)
                raise

        def _new_graph():
            try:
                graph_obj = torch.cuda.CUDAGraph(
                    keep_graph=self._verbose_enabled()
                )
            except TypeError:
                graph_obj = torch.cuda.CUDAGraph()
            if self._verbose_enabled():
                enable_debug = getattr(graph_obj, "enable_debug_mode", None)
                if callable(enable_debug):
                    try:
                        enable_debug()
                    except Exception as exc:
                        self._verbose_exception(
                            label,
                            "enable debug mode failed",
                            exc,
                        )
            return graph_obj

        graph = _new_graph()
        graph_pool = None
        try:
            graph_pool = self.graph_pool()
            graph_cm = (
                torch.cuda.graph(graph, pool=graph_pool)
                if graph_pool is not None
                else torch.cuda.graph(graph)
            )
            with graph_cm, torch.no_grad():
                static_output = fn(*static_args, **static_kwargs)
        except Exception as exc:
            retry_private = (
                graph_pool is not None
                and "use_count > 0" in str(exc)
            )
            if not retry_private:
                self._verbose_exception(label, "capture body/end failed", exc)
                self._cleanup_failed_capture(graph, device, label)
                raise
            self._verbose_exception(
                label,
                "shared-pool capture failed; retrying private pool",
                exc,
            )
            self._cleanup_failed_capture(graph, device, label)
            graph = _new_graph()
            try:
                with torch.cuda.graph(graph), torch.no_grad():
                    static_output = fn(*static_args, **static_kwargs)
            except Exception as retry_exc:
                self._verbose_exception(
                    label,
                    "private-pool retry failed",
                    retry_exc,
                )
                self._cleanup_failed_capture(graph, device, label)
                raise retry_exc from exc
        try:
            instantiate = getattr(graph, "instantiate", None)
            if self._verbose_enabled() and callable(instantiate):
                instantiate()
            graph.replay()
        except Exception as exc:
            self._verbose_exception(label, "initial replay failed", exc)
            self._cleanup_failed_capture(graph, device, label)
            raise
        if self._verbose_enabled():
            try:
                torch.cuda.synchronize(device)
            except Exception as exc:
                self._verbose_exception(label, "post-capture synchronize failed", exc)
                raise
            node_count, dump_path = self._debug_graph_summary(graph, label)
            elapsed = time.perf_counter() - capture_start
            suffix = f" debug_dump={dump_path}" if dump_path is not None else ""
            self._verbose_log(
                label,
                "capture end "
                f"started_at={capture_start_wall:.6f} "
                f"elapsed={elapsed:.6f}s graph_nodes={node_count}{suffix}",
            )
        return _CUDAGraphEntry(
            graph=graph,
            static_args=static_args,
            static_kwargs=static_kwargs,
            static_output=static_output,
            keepalive=tuple(keepalive),
        )


def _replay_lane_kl_totals(
    stacked: torch.Tensor,
    ref_log_probs,
    *,
    full_sequence_kl: bool,
) -> torch.Tensor:
    """Per-lane KL totals for a lane-replay batch of student logits.

    ``stacked`` is ``[lanes, N, L, V]`` student logits, N = total calibration
    rows in replay order. ``ref_log_probs`` is the teacher list, possibly
    regrouped into ``[mb, L, V]`` microbatches by the
    ``calib_microbatch_size > 1`` regrouping at the top of
    ``measure_lane_batched_kl_deltas``. That regrouping concatenates refs on
    dim 0 in calibration order, so consuming exactly each teacher entry's row
    count of student rows at a running offset restores the exact per-row
    pairing for any microbatch size — the same regrouping-aware pairing the
    override-set replay branch uses via a single ``torch.cat`` (kept per-entry
    here so full-sequence teachers never materialize ``[N, L, V]`` at once).
    Row-count mismatches raise instead of silently broadcasting teacher group
    i against student row i and mis-normalizing by N (audit M10).

    Returns fp32 ``[lanes]``: the sum over calibration rows of the
    mean-over-position KL(teacher || student). The caller divides by the total
    calibration row count.
    """
    lane_count = int(stacked.size(0))
    n_rows = int(stacked.size(1))
    kl_totals = torch.zeros(
        lane_count, device=stacked.device, dtype=torch.float32,
    )
    row = 0
    for entry_idx, teacher in enumerate(ref_log_probs):
        if not isinstance(teacher, torch.Tensor):
            raise RuntimeError(
                "lane-replay KL requires tensor ref_log_probs entries; entry "
                f"{entry_idx} is {type(teacher).__name__}"
            )
        teacher = teacher.to(stacked.device).float()
        if teacher.dim() == 2:
            # A single row's distribution stored without the batch dim.
            teacher = teacher.unsqueeze(0)
        if not full_sequence_kl:
            teacher = teacher[:, -1:, :]
        rows = int(teacher.size(0))
        if row + rows > n_rows:
            raise RuntimeError(
                "lane-replay KL teacher/student row mismatch: teacher entry "
                f"{entry_idx} carries {rows} rows starting at student row "
                f"{row}, but the replay produced only {n_rows} student rows "
                "per lane; the ref_log_probs microbatch regrouping does not "
                "match the replay cache's calibration rows"
            )
        student_log_probs = F.log_softmax(
            stacked[:, row:row + rows].float(), dim=-1,
        )
        teacher_probs = teacher.exp().unsqueeze(0)
        kl_per_pos = (
            teacher_probs * (teacher.unsqueeze(0) - student_log_probs)
        ).sum(dim=-1)
        # kl_per_pos: [lanes, rows, L] -> mean over positions per row, then
        # sum the per-row (per calibration sample) KLs into the totals.
        kl_totals += kl_per_pos.mean(
            dim=tuple(range(2, kl_per_pos.dim())),
        ).sum(dim=1)
        row += rows
    if row != n_rows:
        raise RuntimeError(
            f"lane-replay KL consumed {row} teacher rows but the replay "
            f"produced {n_rows} student rows per lane; ref_log_probs does "
            "not cover the calibration set"
        )
    return kl_totals


_KL_CUDA_GRAPH_REGISTRY = CUDAGraphRegistry(
    label="assignment-kl",
    max_entries=4,
    max_entries_env="PRISMAQUANT_KL_CUDA_GRAPH_CACHE_SIZE",
    verbose_env="PRISMAQUANT_KL_CUDA_GRAPHS_VERBOSE",
)


def assignment_hash(assignment: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(name), str(fmt)) for name, fmt in assignment.items()))


_FROZEN_WEIGHT_CACHE_MEMORY_NOTICE_EMITTED = False


def _maybe_disable_frozen_weight_cache_for_memory(
    device: torch.device,
    enabled: bool,
) -> bool:
    """Disable whole-assignment frozen weight caching under tight memory."""
    global _FROZEN_WEIGHT_CACHE_MEMORY_NOTICE_EMITTED
    if not enabled or device.type != "cuda" or not torch.cuda.is_available():
        return enabled
    budget = max_gpu_memory_bytes(device)
    info = cuda_memory_info(device)
    if budget is None or info is None:
        return enabled
    free_bytes, total_bytes = info
    used_bytes = total_bytes - free_bytes
    reserve_frac = float(os.environ.get(
        "PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_FRACTION", "0.05"))
    projected_reserved = used_bytes + int(float(budget) * reserve_frac)
    if projected_reserved < budget:
        return enabled
    if not _FROZEN_WEIGHT_CACHE_MEMORY_NOTICE_EMITTED:
        print(
            "[kl-measurement] disabling frozen weight cache for "
            f"assignment KL: used={used_bytes / 1024 ** 3:.2f}GiB "
            f"budget={budget / 1024 ** 3:.2f}GiB",
            flush=True,
        )
        _FROZEN_WEIGHT_CACHE_MEMORY_NOTICE_EMITTED = True
    return False


def _move_tensor_tree_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        non_blocking = bool(value.device.type == "cpu" and value.is_pinned())
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            key: _move_tensor_tree_to_device(child, device)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree_to_device(child, device) for child in value)
    if isinstance(value, list):
        return [_move_tensor_tree_to_device(child, device) for child in value]
    return value


def _prepare_kl_tensor_inputs(
    calib_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(calib_ids, torch.Tensor):
        return calib_ids
    if torch.device(device).type == "cuda" and calib_ids.device != device:
        if (
            calib_ids.device.type == "cpu"
            and torch.cuda.is_available()
            and not calib_ids.is_pinned()
        ):
            try:
                calib_ids = calib_ids.pin_memory()
            except RuntimeError:
                pass
        non_blocking = bool(calib_ids.device.type == "cpu" and calib_ids.is_pinned())
        return calib_ids.to(device, non_blocking=non_blocking)
    return calib_ids


def _prepare_ref_log_probs_for_kl(ref_log_probs, device: torch.device):
    if torch.device(device).type != "cuda":
        return ref_log_probs
    return _move_tensor_tree_to_device(ref_log_probs, device)


@torch.no_grad()
def sequence_token_nll(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    chunk: int = 128,
) -> float | None:
    """Mean next-token NLL (nats) for one sequence, from logits already in hand.

    R9's "free rung-2 term": the student logits and the calibration ids are both
    live at every KL site, so a PPL-family statistic costs one ``gather`` and a
    ``logsumexp`` — no extra forward.  Returns ``None`` when there is no
    next-token label to score, which is the last-token KL scope (the model only
    emitted position ``T-1``, whose label lies outside the window).

    Chunked over positions so the fp32 upcast stays bounded: a full
    ``[T, V]`` fp32 copy on a 150k vocab is ~300 MB at T=512.
    """
    if logits.dim() != 3 or logits.size(0) != 1:
        return None
    if logits.size(1) < 2:
        return None
    pred = logits[0, :-1, :]
    targets = token_ids.reshape(-1)[1:1 + pred.size(0)].to(pred.device)
    if targets.numel() != pred.size(0):
        return None
    targets = targets.to(torch.int64)
    total = 0.0
    count = 0
    step = max(int(chunk), 1)
    for start in range(0, pred.size(0), step):
        block = pred[start:start + step].float()
        lse = torch.logsumexp(block, dim=-1)
        picked = block.gather(
            1, targets[start:start + step].unsqueeze(1)).squeeze(1)
        total += float((lse - picked).sum().item())
        count += block.size(0)
    if count == 0:
        return None
    return total / count


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile, matching ``torch.Tensor.quantile``.

    Kept identical to the gold lane's reduction (``tools/measure_vllm_full_kl``
    uses ``tensor.quantile``) so a selection row and a served row are read the
    same way.
    """
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("quantile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    pos = float(q) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize_per_sequence_kl(
    kl_values: Sequence[float],
    *,
    nll_values: Sequence[float] | None = None,
) -> dict[str, object]:
    """Tail statistics over per-sequence KL (and, when available, token NLL).

    Key names are deliberately the gold lane's (`tools/measure_vllm_full_kl.py`
    emits ``kl_mean``/``kl_p99``/``kl_max``/``kl_per_sample``) so a selection row
    and a served row are directly comparable.  ``kl_tail_domain`` records the
    honest difference: here the sample unit is a **sequence** (each value is
    already a position-mean from ``kl_divergence``), not a position, so
    ``kl_p99`` is a p99 over sequences.
    """
    vals = [float(v) for v in kl_values]
    if not vals:
        raise ValueError("per-sequence KL summary received no values")
    out: dict[str, object] = {
        "kl_per_sample": vals,
        "kl_p95": _quantile(vals, 0.95),
        "kl_p99": _quantile(vals, 0.99),
        "kl_max": max(vals),
        "kl_tail_domain": "sequence",
    }
    nlls = [float(v) for v in (nll_values or []) if v is not None and math.isfinite(float(v))]
    if nlls:
        out["nll_per_sample"] = nlls
        out["nll_mean"] = sum(nlls) / len(nlls)
        out["nll_p99"] = _quantile(nlls, 0.99)
    return out


@torch.no_grad()
def measure_assignment_kl(
    model,
    assignment: Mapping[str, str],
    calib_ids: torch.Tensor,
    ref_log_probs,
    *,
    work_root: str | Path,
    profile=None,
    perturbed_cache: PerturbedActivationCache | None = None,
    use_frozen_weight_cache: bool = True,
    production_weight_cache=None,
    rng_seed: int | None = 0,
    kl_scope: KLScope | None = None,
    include_activation_quant: bool = True,
    stream_ref_log_probs: bool = False,
    use_cuda_graphs: bool | None = None,
    return_per_sequence: bool = False,
) -> float | tuple[float, list[float], dict[str, object]]:
    """Measure assignment KL on the production perturbed-weight path.

    ``return_per_sequence`` (R9) switches the return to
    ``(mean, per_sequence_kl, stats)`` and additionally computes the free
    rung-2 token NLL from the student logits already in hand — ``stats``
    carries ``nll_per_sample`` (``None`` under the last-token scope, which has
    no next-token label).  Default ``False`` keeps the historical scalar return
    and does no extra work, so every existing caller is byte-identical.
    """
    device = next(model.parameters()).device
    calib_ids = _prepare_kl_tensor_inputs(calib_ids, device)
    if not stream_ref_log_probs:
        ref_log_probs = _prepare_ref_log_probs_for_kl(ref_log_probs, device)
    effective_kl_scope = resolve_kl_scope(kl_scope)
    if use_frozen_weight_cache and not _env_flag_enabled(
        "PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE",
        default=True,
    ):
        use_frozen_weight_cache = False
    use_frozen_weight_cache = _maybe_disable_frozen_weight_cache_for_memory(
        device, use_frozen_weight_cache)
    hooks = perturbed_cache
    cal_hash = calibration_data_hash(calib_ids)
    if hooks is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="prismaquant_kl_hooks_", dir=str(work_root)))
        hooks = PerturbedActivationCache(
            model,
            assignment,
            cache_dir,
            input_rows=0,
            cal_hash=cal_hash,
            profile=profile,
            production_weight_cache=production_weight_cache,
            include_activation_quant=include_activation_quant,
        )
        strict_coverage_default = (
            production_weight_cache is not None
            or _env_flag_enabled(
                "PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT",
                default=False,
            )
        )
        if _env_flag_enabled(
            "PRISMAQUANT_STRICT_ASSIGNMENT_COVERAGE",
            default=strict_coverage_default,
        ):
            missing = [
                name for name in hooks.missing
                if fr.canonical_format_name(assignment.get(name, "BF16"))
                != "BF16"
            ]
            if missing:
                raise RuntimeError(
                    "assignment contains non-BF16 qnames that do not "
                    "resolve on the live model; refusing to measure a "
                    f"partial assignment.  missing={len(missing)} "
                    f"sample={missing[:5]}"
                )
            if hooks.skipped:
                raise RuntimeError(
                    "assignment has conflicting activation-quant formats "
                    "within at least one module; refusing to measure with "
                    f"activation quant silently skipped.  sample="
                    f"{hooks.skipped[:3]}"
                )
        # A unit whose spec is measured under the served static-scale
        # activation contract (a Tessera W4A4 rung) is priced through the
        # served oracle at ITS calibrated G, or not at all: the hook refuses
        # by name, and this asks the same question before the first forward
        # so the refusal lists every such unit rather than the first one the
        # model reaches.  Not gated by the strict-coverage lever -- there is
        # no dynamic serving path to fall back to (#205).
        scale_gaps = hooks.served_activation_scale_gaps()
        if scale_gaps:
            raise ActivationScaleContractError(
                "assignment-KL: these units are served under the static "
                "activation contract but the hooks have no calibrated "
                "activation maximum for them (production cache "
                "activation_max_abs); refusing to measure them under a "
                f"dynamic quantiser the runtime never runs.  missing="
                f"{len(scale_gaps)} sample={scale_gaps[:5]}"
            )
    values = []
    nll_values: list[float] = []
    if use_cuda_graphs is None:
        use_cuda_graphs = _env_cuda_graphs_enabled_for_call_count(
            "PRISMAQUANT_KL_CUDA_GRAPHS",
            default="auto",
            call_count=int(calib_ids.size(0)),
            min_calls=16,
        )
    else:
        use_cuda_graphs = bool(use_cuda_graphs)
    graph_key = (
        id(model),
        assignment_hash(assignment),
        bool(use_frozen_weight_cache),
        effective_kl_scope,
        bool(include_activation_quant),
        rng_seed,
        cal_hash,
        id(production_weight_cache) if production_weight_cache is not None else 0,
    )
    cache_cm = nullcontext()
    if use_frozen_weight_cache and hooks._frozen_weight_cache is None:
        cache_cm = hooks.frozen_weight_cache()
    rng_devices = []
    if rng_seed is not None and device.type == "cuda" and torch.cuda.is_available():
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    rng_cm = (
        torch.random.fork_rng(devices=rng_devices)
        if rng_seed is not None else nullcontext()
    )
    installed_here = not hooks.installed
    with cache_cm:
        materialized_cm = nullcontext()
        if (
            use_cuda_graphs
            and use_frozen_weight_cache
            and device.type == "cuda"
            and torch.cuda.is_available()
            and hooks._frozen_weight_cache is not None
        ):
            materialized_cm = hooks.materialized_frozen_weights()
        with materialized_cm:
            if installed_here:
                hooks.install()
            try:
                full_seq = effective_kl_scope == "full_sequence"
                with rng_cm:
                    if rng_seed is not None:
                        torch.manual_seed(int(rng_seed))
                        if device.type == "cuda" and torch.cuda.is_available():
                            torch.cuda.manual_seed_all(int(rng_seed))
                    for i in range(calib_ids.size(0)):
                        batch = calib_ids[i:i + 1].to(device)
                        if full_seq:
                            def _forward(batch_ids):
                                return model(batch_ids).logits.clone()
                        else:
                            def _forward(batch_ids):
                                return model(batch_ids).logits[:, -1:, :].clone()

                        logits = _KL_CUDA_GRAPH_REGISTRY.run(
                            "assignment-kl-forward",
                            graph_key,
                            _forward,
                            batch,
                            enabled=use_cuda_graphs,
                            device=device,
                            keepalive=(hooks,),
                        )
                        teacher = ref_log_probs[i] if full_seq else ref_log_probs[i][:, -1:, :]
                        teacher = _move_tensor_tree_to_device(teacher, device)
                        # Hard gate: kl_divergence broadcasts, so a teacher
                        # built at a different KL scope than the student (e.g.
                        # a last-token [1,1,V] reference meeting a
                        # full-sequence [1,T,V] student because
                        # PRISMAQUANT_FULL_SEQUENCE_KL resolved the scope)
                        # would silently produce mean_t KL(p_last || q_t) —
                        # not a KL of anything (audit M7).
                        if (
                            isinstance(teacher, torch.Tensor)
                            and tuple(teacher.shape) != tuple(logits.shape)
                        ):
                            raise RuntimeError(
                                "assignment-KL teacher/student shape mismatch: "
                                f"teacher {tuple(teacher.shape)} vs student "
                                f"{tuple(logits.shape)} at sample {i} "
                                f"(kl_scope={effective_kl_scope!r}). "
                                "ref_log_probs must be built at the same KL "
                                "scope the measurement resolves to; pass "
                                "kl_scope explicitly at the call site that "
                                "built the references instead of relying on "
                                "PRISMAQUANT_FULL_SEQUENCE_KL."
                            )
                        values.append(float(kl_divergence(logits, teacher).item()))
                        if return_per_sequence and full_seq:
                            nll = sequence_token_nll(logits, batch)
                            if nll is not None:
                                nll_values.append(nll)
            finally:
                if installed_here:
                    hooks.remove()
    mean = sum(values) / max(len(values), 1)
    if not return_per_sequence:
        return mean
    stats: dict[str, object] = {
        "mode": "hooks",
        "cuda_graphs": bool(use_cuda_graphs),
        "kl_scope": effective_kl_scope,
        "n_sequences": len(values),
        "nll_per_sample": nll_values or None,
    }
    return mean, list(values), stats
