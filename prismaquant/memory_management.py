"""Shared memory controls for cache-heavy quantization passes."""
from __future__ import annotations

import gc
import os
import sys
import weakref
from typing import Iterable

import torch


_BUDGET_EVICTORS: "weakref.WeakSet[object]" = weakref.WeakSet()


class GPUMemoryBudgetExceeded(RuntimeError):
    """Raised when cache eviction cannot bring CUDA memory under budget."""


def env_flag_enabled(name: str, *, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return max(parsed, 0)


def register_budget_evictor(evictor: object) -> None:
    try:
        _BUDGET_EVICTORS.add(evictor)
    except TypeError:
        pass


def unregister_budget_evictor(evictor: object) -> None:
    try:
        _BUDGET_EVICTORS.discard(evictor)
    except TypeError:
        pass


def cuda_memory_info(device: torch.device | None = None) -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except TypeError:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    if _use_host_available_for_uma(device):
        host_info = _host_memory_info()
        if host_info is not None:
            host_available, _host_total = host_info
            # Integrated CUDA devices share the host memory pool.  On GB10,
            # mem_get_info() reports reclaimable page cache as unavailable,
            # which makes cache and lane guardrails far too conservative
            # after streamed weight snapshots.  MemAvailable already includes
            # reclaimable cache, so use it as the better free-memory signal.
            free_bytes = max(int(free_bytes), int(host_available))
            free_bytes = min(int(free_bytes), int(total_bytes))
    return int(free_bytes), int(total_bytes)


def _use_host_available_for_uma(device: torch.device | None = None) -> bool:
    mode = os.environ.get("PRISMAQUANT_UMA_MEMORY_INFO", "auto")
    mode = str(mode).strip().lower()
    if mode in {"0", "false", "no", "off", "cuda"}:
        return False
    if mode in {"1", "true", "yes", "on", "host", "uma"}:
        return True
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return False
    return bool(getattr(props, "is_integrated", False))


def _host_memory_info() -> tuple[int, int] | None:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total)
    except Exception:
        pass
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, rest = line.split(":", 1)
                if key in {"MemAvailable", "MemTotal"}:
                    values[key] = int(rest.strip().split()[0]) * 1024
        if "MemAvailable" in values and "MemTotal" in values:
            return values["MemAvailable"], values["MemTotal"]
    except Exception:
        return None
    return None


def _dynamic_gpu_memory_budget_bytes(
    device: torch.device | None = None,
) -> int | None:
    info = cuda_memory_info(device)
    if info is None:
        return None
    free_bytes, total_bytes = info
    used_bytes = total_bytes - free_bytes

    device_reserve = max(
        int(total_bytes * max(
            env_float("PRISMAQUANT_GPU_MEM_RESERVE_FRACTION", 0.05),
            0.0,
        )),
        int(max(env_float("PRISMAQUANT_GPU_MEM_RESERVE_GB", 2.0), 0.0) * 1024 ** 3),
    )
    budget = total_bytes - device_reserve

    host_info = _host_memory_info()
    if host_info is not None:
        host_available, host_total = host_info
        host_reserve = max(
            int(host_total * max(
                env_float("PRISMAQUANT_HOST_MEM_RESERVE_FRACTION", 0.05),
                0.0,
            )),
            int(max(env_float("PRISMAQUANT_HOST_MEM_RESERVE_GB", 4.0), 0.0) * 1024 ** 3),
        )
        host_deficit = max(0, host_reserve - host_available)
        if host_deficit:
            # On UMA systems, CUDA allocations and host memory share the same
            # physical pool. Lower the CUDA cache budget by the observed host
            # deficit so registered caches are evicted before swap pressure
            # turns into a system OOM.
            budget = min(budget, used_bytes - host_deficit)

    return max(int(budget), 0)


def max_gpu_memory_bytes(device: torch.device | None = None) -> int | None:
    """Return the cache budget for CUDA-visible allocations.

    `PRISMAQUANT_MAX_GPU_MEM_GB` remains an explicit override. Without it,
    derive the budget from the live device size and host memory pressure so
    cache-heavy passes scale across 24 GB, 48 GB, 96 GB, UMA, and larger hosts
    without baking in one workstation's usable-memory ceiling.
    """
    raw = os.environ.get("PRISMAQUANT_MAX_GPU_MEM_GB")
    if raw is not None and raw.strip() != "":
        try:
            gb = float(raw)
        except ValueError:
            gb = -1.0 if raw.strip().lower() in {"0", "off", "false", "none"} else 0.0
        if gb <= 0.0:
            return None
        return int(gb * 1024 ** 3)
    return _dynamic_gpu_memory_budget_bytes(device)


def _gb(num_bytes: int | float) -> float:
    return float(num_bytes) / float(1024 ** 3)


def _tensor_tree_nbytes(value, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key in seen:
            return 0
        seen.add(key)
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_tree_nbytes(child, seen) for child in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_tree_nbytes(child, seen) for child in value)
    return 0


def _graph_entry_static_nbytes(entry) -> int:
    seen: set[int] = set()
    total = 0
    for attr in ("static_hidden", "static_args", "static_kwargs", "static_output"):
        if hasattr(entry, attr):
            total += _tensor_tree_nbytes(getattr(entry, attr), seen)
    return total


def _evictor_pool_id(evictor: object) -> str:
    getter = getattr(evictor, "graph_pool_id", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception as exc:
            return f"unavailable:{type(exc).__name__}"
    pool = getattr(evictor, "graph_pool", None)
    if callable(pool):
        try:
            handle = pool()
        except Exception as exc:
            return f"unavailable:{type(exc).__name__}"
        if handle is None:
            return "private"
        return f"shared:{id(handle):x}"
    return "unknown"


def report_graph_memory(label: str = "") -> None:
    """Print per-registry occupancy + pool footprint. Used at phase boundaries."""
    if not env_flag_enabled("PRISMAQUANT_GRAPH_AUDIT", default=False):
        return
    label_text = str(label or "-")
    evictors = list(_BUDGET_EVICTORS)
    registry_count = 0
    total_entries = 0
    total_static_bytes = 0
    for evictor in evictors:
        entries = getattr(evictor, "entries", None)
        if entries is None:
            continue
        try:
            entry_items = list(entries.values())
        except AttributeError:
            continue
        registry_count += 1
        entry_count = len(entry_items)
        static_bytes = sum(_graph_entry_static_nbytes(entry) for entry in entry_items)
        total_entries += entry_count
        total_static_bytes += static_bytes
        registry_label = getattr(evictor, "label", type(evictor).__name__)
        print(
            "[graph-audit] "
            f"label={label_text} registry={registry_label} "
            f"class={type(evictor).__name__} entries={entry_count} "
            f"static_bytes={static_bytes} static_gb={_gb(static_bytes):.6f} "
            f"pool={_evictor_pool_id(evictor)}",
            file=sys.stderr,
            flush=True,
        )

    allocated_current = None
    mem_info = None
    if torch.cuda.is_available():
        try:
            allocated_current = int(
                torch.cuda.memory_stats().get("allocated_bytes.all.current", 0)
            )
        except Exception:
            allocated_current = None
        mem_info = cuda_memory_info()
    free_text = total_text = "n/a"
    if mem_info is not None:
        free_text = str(mem_info[0])
        total_text = str(mem_info[1])
    allocated_text = "n/a" if allocated_current is None else str(allocated_current)
    print(
        "[graph-audit] "
        f"label={label_text} summary registries={registry_count} "
        f"entries={total_entries} static_bytes={total_static_bytes} "
        f"static_gb={_gb(total_static_bytes):.6f} "
        f"allocated_bytes_current={allocated_text} "
        f"mem_free_bytes={free_text} mem_total_bytes={total_text}",
        file=sys.stderr,
        flush=True,
    )


def _unique_evictors(evictors: Iterable[object]) -> list[object]:
    out: list[object] = []
    seen: set[int] = set()
    for evictor in evictors:
        if evictor is None:
            continue
        key = id(evictor)
        if key in seen:
            continue
        seen.add(key)
        out.append(evictor)
    return out


def _drop_released_cuda_memory(*, synchronize: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if synchronize:
            torch.cuda.synchronize()


def enforce_gpu_memory_budget(
    evictors: Iterable[object] = (),
    *,
    device: torch.device | None = None,
    reason: str = "",
) -> int:
    """Evict oldest registered cache entries until used CUDA memory is in budget.

    ``torch.cuda.mem_get_info`` reports driver-visible free memory. We compare
    ``total - free`` against ``PRISMAQUANT_MAX_GPU_MEM_GB`` so the budget acts
    as a hard ceiling even when PyTorch's caching allocator is holding blocks.
    """
    budget_bytes = max_gpu_memory_bytes(device)
    if budget_bytes is None:
        return 0
    info = cuda_memory_info(device)
    if info is None:
        return 0
    free_bytes, total_bytes = info
    used_bytes = total_bytes - free_bytes
    if used_bytes <= budget_bytes:
        return 0

    candidates = _unique_evictors([*evictors, *_BUDGET_EVICTORS])
    evicted = 0
    while used_bytes > budget_bytes:
        progress = False
        for evictor in candidates:
            evict_one = getattr(evictor, "evict_oldest_for_memory_budget", None)
            if not callable(evict_one):
                continue
            if evict_one():
                evicted += 1
                progress = True
                _drop_released_cuda_memory()
                info = cuda_memory_info(device)
                if info is None:
                    return evicted
                free_bytes, total_bytes = info
                used_bytes = total_bytes - free_bytes
                if used_bytes <= budget_bytes:
                    break
        if not progress:
            detail = f" during {reason}" if reason else ""
            raise GPUMemoryBudgetExceeded(
                "CUDA memory budget exceeded"
                f"{detail}: used={_gb(used_bytes):.2f}GB "
                f"budget={_gb(budget_bytes):.2f}GB "
                f"total={_gb(total_bytes):.2f}GB. "
                "No registered cache entries remain to evict; the model or "
                "other allocations exceed PRISMAQUANT_MAX_GPU_MEM_GB."
            )
    return evicted


def phase_boundary_memory_cleanup(label: str | None = None) -> None:
    """Release allocator-held memory and collect Python garbage at phase edges."""
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        if label:
            print(
                f"[memory] cleanup {label}: empty_cache failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
