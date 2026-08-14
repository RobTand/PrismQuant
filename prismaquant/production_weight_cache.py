"""Production-faithful rendered-weight cache for measured candidates.

Per-Linear candidate probes and real-KL gates need to measure the same
rendered weights that export will ship. Without this cache the perturbation
installed into the model is bare RTN. The export pipeline renders weights with
several activation-aware passes; the shipped δw is much smaller than the RTN
δw at the same format.

This module pre-renders `W_tilde[name, fmt]` once, using the production
quantization path:

  IMPLEMENTED (v1):
    * scalar GPTQ (with damp sweep when env-enabled)
    * optional scalar scale-sweep
    * joint NVFP4 fused-sibling globals (q/k/v share a per-tensor scale,
      gate/up share theirs)
    * calibrated `input_global_scale` per fused-sibling group
      (max_abs(activations) / 6.0; the same value the export persists
      to the artifact)
    * progressive local render gates using the shared mechanism order
      baseline -> format scale rule -> GPTQ -> optional scale_sweep;
      individual formats explicitly opt out of unsupported mechanisms,
      and regressive candidates fall back to the previous accepted render
      while recording metadata
    * FP8_DYNAMIC/FP8_E4M3 per-row scale search when scale_sweep is enabled;
      explicit MXFP8 E8M0 scale search remains opt-in. These refine the current
      accepted render rather than starting a separate format-specific path
    * activation-weighted GPTQ for FP8_DYNAMIC/FP8_E4M3. Explicit MXFP8 keeps
      GPTQ support for research/legacy artifacts. NVFP4 is the only production
      format that uses joint_scale_opt; MXFP8 uses the canonical E8M0 scale rule.
    * retired Fisher-weighted local objectives are archived under
      ``archive/fisher_2026-05-15/`` and are not part of the production
      pipeline
    * retired input-axis rotation experiments are not part of the production
      cache path.

  KNOWN GAPS (v2 work, NOT implemented):
    * batched NVFP4 GPTQ + scale-sweep across same-shape Linears
      (defaults-on in the export when activations are cached;
      mathematically equivalent to scalar but ~3-8× faster on MoE)
    * block-output match (post-GPTQ refinement against BF16 block output)
    * any export-only refinements added after this docstring is written

  FP8_DYNAMIC / BF16:
    * FP8_DYNAMIC is represented by the canonical FP8_E4M3 format name:
      per-output-row FP32 weight scales and per-token dynamic activation
      scales. It uses GPTQ damp-sweep by default in production render.
    * Explicit MXFP8/MXFP4 formats remain available only when requested.
    * BF16 is passthrough.

PerturbedActivationCache installs `W_tilde` (and applies the calibrated
`input_global_scale` on activations) instead of RTN-quantizing on the
fly, so per-Linear probes, frontier validation, and polish gates use the
same δw the export will deliver, modulo the v2 gaps above.

Usage:

    cache = fill_production_weight_cache(
        model, calib_ids, qnames=qnames, formats=["NVFP4"],
    )
    cache.validate_coverage(qnames, ["NVFP4"])  # raise on misses
    perturbed_cache = PerturbedActivationCache(
        ..., production_weight_cache=cache,
    )
"""
from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re

import torch
import torch.nn as nn

from prismaquant.activation_sampling import update_priority_reservoir
from prismaquant.build_rtn_cache import iter_quantizable_tensors
from prismaquant.render_score import (
    gate_render_candidate,
    normalize_row_weights,
    resolve_render_mechanism_order,
    score_render_error,
)
from prismaquant.source_prefetch import prefetch_files_to_page_cache


CB_RENDER_IDENTITY_SCHEMA = (
    "prismaquant.production_weight_cache.cb_render_identity.v2"
)
CB_COL_WEIGHTS_HASH_SCHEMA = (
    "prismaquant.production_weight_cache.cb_col_weights.v1"
)
CB_SOURCE_WEIGHTS_HASH_SCHEMA = (
    "prismaquant.production_weight_cache.cb_source_weights.v1"
)
CB_RENDER_CONTRACT_SCHEMA = (
    "prismaquant.production_weight_cache.cb_render_contract.v1"
)
CB_RENDER_MECHANISM_ABI = "prismaquant.production_render_mechanisms.v1"
CB_RENDER_IDENTITY_METADATA_KEY = "cb_render_identity"


def _render_base_format(fmt: str) -> str:
    return str(fmt).strip().upper()


def _cache_weight_filename(qname: str, fmt: str) -> str:
    safe = qname.replace("/", "__").replace(".", "_")
    return f"{safe}__{fmt}.pt"


_UNCACHED_PACKED_EXPERT_RE = re.compile(
    r"\.experts(?:\.\d+)?\."
    r"(?:gate_up_proj|down_proj|gate_proj|up_proj|w1|w2|w3)$"
)


def is_uncached_packed_expert_qname(qname: str) -> bool:
    """Return True for packed-MoE expert tensor qnames.

    Legacy residency helpers may skip these names when no concrete packed
    render is expected. Production build/export gates must be stricter:
    non-BF16 packed experts are rendered by
    ``fill_packed_expert_cache_entries`` and missing entries are an early
    coverage failure unless the explicit RTN research escape hatch is set.
    """
    return bool(_UNCACHED_PACKED_EXPERT_RE.search(str(qname)))


_PACKED_EXPERT_PARAM_RE = re.compile(
    r"\.experts\."
    r"(?:gate_up_proj|down_proj|gate_proj|up_proj|w1|w2|w3)$"
)


def is_packed_expert_param_qname(qname: str) -> bool:
    """Return True only for PACKED (3-D, un-indexed) expert param qnames.

    Stricter than ``is_uncached_packed_expert_qname``, which also matches
    per-expert-indexed names (``...experts.5.down_proj``) — those are plain
    nn.Linear modules (DSv4/MiniMax layouts) whose activation replay capture
    IS per-projection-correct. Use this predicate when a behavior must apply
    only to packed tensors, where one module plan carries several projection
    params and the module-input capture is the wrong tensor for ``down``.
    """
    return bool(_PACKED_EXPERT_PARAM_RE.search(str(qname)))


@dataclass
class ProductionWeightCache:
    """Dict-like cache of production-faithful dequantized weights.

    Keys: ``(qname, fmt_canonical)``.  Values are EITHER:
      * ``torch.Tensor`` ([out, in] float32 or bf16) — in-memory cache
      * ``str`` (a path) — points to a per-Linear .pt file on disk;
        ``get()`` lazy-loads on first access and memoizes the tensor

    Disk-streaming mode (when ``cache_dir`` is set during fill) keeps
    fill-time peak memory bounded — only one weight is in RAM at a time
    instead of the full ~25 GB stack of all rendered Linears.  At
    validation or recache time the lazy-load caches each weight in memory
    after first access, so steady-state behavior matches the in-memory mode.

    ``activation_max_abs[qname]`` is the calibrated max(|activations|)
    used by the act-clip step in the export pipeline.  PerturbedActivation
    Cache reads this and clamps activations to ``[-max_abs, +max_abs]``
    before per-group RTN, matching the export's act-clip behavior.

    Note: the *exported metadata* convention for this field is
    ``input_global_scale = 448 * 6 / max_abs`` (reciprocal — vLLM
    multiplies activations by it; legacy ``6 / max_abs`` under
    PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE=0).  We store ``max_abs``
    directly here because that's the value the act-clip path needs;
    consumers convert via
    ``export_native_compressed._nvfp4_input_global_scale_from_max_abs``.
    """
    weights: dict[tuple[str, str], object]  # tensor OR str(path)
    levers: dict[str, bool]
    activation_max_abs: dict[str, float] | None = None
    failed: dict[tuple[str, str], str] | None = None
    cache_dir: str | None = None  # set when disk-streaming was used at fill time
    metadata: dict[str, object] | None = None
    # Backward-compat alias for code that still reads ``activation_scales``.
    activation_scales: dict[str, float] | None = None
    # LRU eviction state for memoized tensor loads.  When non-None, the
    # in-memory cache holds at most ``mem_lru_max_bytes`` of tensor data;
    # least-recently-used entries are evicted back to their on-disk
    # filename when the budget is exceeded.  Default OFF for backward
    # compat; opt-in via ``enable_lru(...)``.
    _lru_order: list[tuple[str, str]] | None = None
    _lru_paths: dict[tuple[str, str], str] | None = None
    _lru_bytes: int = 0
    _lru_max_bytes: int = 0

    def __post_init__(self) -> None:
        # Normalize to ``activation_max_abs`` if a caller used the legacy
        # name.  After this, both attributes hold the same dict (max_abs).
        if self.activation_max_abs is None and self.activation_scales is not None:
            self.activation_max_abs = self.activation_scales
        elif self.activation_scales is None and self.activation_max_abs is not None:
            self.activation_scales = self.activation_max_abs

    def validate_cb_render_identity(
        self,
        *,
        expected_context=None,
        expected_qnames: Sequence[str] | None = None,
        col_weights: Mapping[str, torch.Tensor] | None = None,
        require_for_formats: Sequence[str] = (),
        require_source_complete: bool = True,
        where: str = "ProductionWeightCache",
    ):
        """Validate and return this cache's persisted CB producer context.

        The context is read from the cache metadata, never reconstructed from
        the current process environment.  Non-CB caches return ``None``.
        """
        return validate_production_cache_cb_render_identity(
            self,
            expected_context=expected_context,
            expected_qnames=expected_qnames,
            col_weights=col_weights,
            require_for_formats=require_for_formats,
            require_source_complete=require_source_complete,
            where=where,
        )

    def enable_lru(self, max_bytes: int) -> None:
        """Bound the in-memory tensor footprint to ``max_bytes`` via LRU
        eviction.  Required for very large disk-streamed caches (e.g.
        Qwen3.6-27B's ~46 GB of bf16 weights wouldn't fit in a 121 GB
        UMA box alongside the model + working set)."""
        self._lru_max_bytes = int(max_bytes)
        self._lru_order = []
        self._lru_paths = {}
        self._lru_bytes = 0

    def _evict_to_budget(self) -> None:
        if self._lru_order is None or self._lru_max_bytes <= 0:
            return
        while self._lru_bytes > self._lru_max_bytes and self._lru_order:
            evict_key = self._lru_order.pop(0)
            t = self.weights.get(evict_key)
            if isinstance(t, torch.Tensor):
                self._lru_bytes -= t.element_size() * t.numel()
                # Restore the filename so subsequent lookups still resolve.
                if self._lru_paths is not None and evict_key in self._lru_paths:
                    self.weights[evict_key] = self._lru_paths[evict_key]

    def compact_for_pickle(self) -> int:
        """Restore disk-backed resident tensors to path references.

        A recache/polish pass may lazily load many disk-streamed entries into
        ``weights``.  Pickling that state would serialize the tensors and turn a
        small manifest into a multi-GB file.  This method keeps the cache
        portable by replacing resident LRU-loaded tensors with their original
        paths before serialization.  Returns the number of entries compacted.
        """
        compacted = 0
        for key, path in (self._lru_paths or {}).items():
            if isinstance(self.weights.get(key), torch.Tensor):
                self.weights[key] = path
                compacted += 1
        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
            for key, value in list(self.weights.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                fname = _cache_weight_filename(key[0], key[1])
                if (cache_dir / fname).is_file():
                    self.weights[key] = fname
                    compacted += 1
        self._lru_order = [] if self._lru_order is not None else None
        self._lru_bytes = 0
        return compacted

    def _path_for_value(self, value: object) -> str:
        path = str(value)
        if self.cache_dir and not Path(path).is_absolute():
            path = str(Path(self.cache_dir) / path)
        return path

    def _name_candidates(self, name: str) -> list[str]:
        candidates = [name]
        if name.endswith(".weight"):
            candidates.append(name[:-len(".weight")])
        if name.startswith("model.language_model."):
            candidates.append("model." + name[len("model.language_model."):])
        return list(dict.fromkeys(candidates))

    def _format_candidates(self, fmt: str) -> list[str]:
        raw = str(fmt)
        candidates = [raw, raw.upper()]
        try:
            from prismaquant import format_registry as fr
            candidates.append(fr.canonical_format_name(raw))
        except Exception:
            pass
        if "MXFP8_E4M3" in candidates:
            candidates.append("MXFP8")
        if "MXFP8" in candidates:
            candidates.append("MXFP8_E4M3")
        if "FP8_E4M3" in candidates:
            candidates.append("FP8")
        if "FP8" in candidates:
            candidates.append("FP8_E4M3")
        return list(dict.fromkeys(candidates))

    def resolve_key(self, name: str, fmt: str) -> tuple[str, str] | None:
        """Resolve recipe aliases to the concrete stored cache key."""
        for cand in self._name_candidates(name):
            for fmt_cand in self._format_candidates(fmt):
                key = (cand, fmt_cand)
                if key in self.weights:
                    return key
        return None

    def estimate_nbytes(
        self,
        keys: Sequence[tuple[str, str]] | None = None,
    ) -> int:
        """Estimate resident bytes for cache entries without loading them."""
        total = 0
        for key in (list(self.weights) if keys is None else list(keys)):
            value = self.weights.get(key)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                total += value.element_size() * value.numel()
            else:
                total += Path(self._path_for_value(value)).stat().st_size
        return total

    def assignment_keys(
        self,
        assignment: Mapping[str, str],
        *,
        include_packed_experts: bool = False,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Return concrete non-BF16 cache keys needed by an assignment.

        This centralizes recipe alias handling for recache, polish, KL
        probes, and export: callers should ask the cache which stored key a
        recipe entry maps to, then feed those keys into ``prefetch``. Legacy
        residency callers leave ``include_packed_experts`` false because
        packed experts may be handled out-of-band; production build/export
        gates set it true once ``fill_packed_expert_cache_entries`` is
        responsible for those 3-D entries.
        """
        from prismaquant import format_registry as fr

        cb_formats = [
            (str(qname), fr.canonical_format_name(str(fmt)))
            for qname, fmt in assignment.items()
            if _is_cb_format_name(fr.canonical_format_name(str(fmt)))
        ]
        if cb_formats:
            cb_expected = []
            for qname, fmt in cb_formats:
                resolved = self.resolve_key(qname, fmt)
                if resolved is not None:
                    cb_expected.append(resolved[0])
            self.validate_cb_render_identity(
                expected_qnames=cb_expected,
                require_for_formats=[fmt for _qname, fmt in cb_formats],
                where="ProductionWeightCache assignment",
            )

        keys: list[tuple[str, str]] = []
        missing: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for qname, fmt in assignment.items():
            fmt_canon = fr.canonical_format_name(str(fmt))
            if fmt_canon == "BF16":
                continue
            key = self.resolve_key(str(qname), fmt_canon)
            if key is None:
                if (
                    is_uncached_packed_expert_qname(str(qname))
                    and not include_packed_experts
                ):
                    continue
                missing.append((str(qname), fmt_canon))
                continue
            if key not in seen:
                keys.append(key)
                seen.add(key)
        return keys, missing

    def assignment_file_paths(
        self,
        assignment: Mapping[str, str],
    ) -> tuple[list[Path], list[tuple[str, str]], list[tuple[str, str]]]:
        """Return disk files backing an assignment without loading tensors.

        This is a page-cache residency helper for validation paths that
        destructively materialize one assignment into the model. It reuses the
        same cache key resolution as ``prefetch_assignment`` but intentionally
        does not call ``torch.load`` or create another rendered-weight cache.
        """
        keys, missing = self.assignment_keys(assignment)
        paths: list[Path] = []
        in_memory: list[tuple[str, str]] = []
        seen_paths: set[Path] = set()
        for key in keys:
            value = self.weights.get(key)
            if value is None:
                missing.append(key)
                continue
            if isinstance(value, torch.Tensor):
                path_value = (
                    self._lru_paths.get(key)
                    if self._lru_paths is not None else None
                )
                if path_value is None:
                    in_memory.append(key)
                    continue
                value = path_value
            path = Path(self._path_for_value(value)).resolve()
            if path not in seen_paths:
                paths.append(path)
                seen_paths.add(path)
        return paths, missing, in_memory

    def prefetch_assignment_file_pages(
        self,
        assignment: Mapping[str, str],
        *,
        mode: str = "require",
        max_resident_bytes: int | None = None,
        headroom_gb: float = 24.0,
        max_workers: int = 4,
        progress: bool = False,
        log_prefix: str = "[prod-cache-files]",
    ) -> dict[str, object]:
        """Prefetch assignment cache files into the OS page cache.

        Unlike ``prefetch_assignment``, this keeps rendered weights out of the
        Python heap. The following ``get`` calls still go through
        ``ProductionWeightCache`` and its LRU, but deserialization reads from
        resident file pages instead of faulting against NVMe.
        """
        paths, missing, in_memory = self.assignment_file_paths(assignment)
        mode = str(mode or "off").lower()
        if paths:
            stats = prefetch_files_to_page_cache(
                paths,
                mode=mode,
                max_resident_bytes=max_resident_bytes,
                headroom_gb=headroom_gb,
                workers=max_workers,
                progress=progress,
                log_prefix=log_prefix,
                label="production cache files",
            )
        else:
            stats = {
                "mode": mode,
                "label": "production cache files",
                "files": 0,
                "bytes": 0,
                "max_resident_bytes": int(max_resident_bytes or 0),
                "available_bytes": None,
                "prefetched_bytes": 0,
                "elapsed_seconds": 0.0,
                "skipped": True,
                "reason": "no disk-backed production cache files",
            }
        stats["keys"] = len(paths) + len(in_memory)
        stats["in_memory"] = len(in_memory)
        stats["missing"] = len(missing)
        if missing:
            stats["missing_sample"] = missing[:8]
            msg = (
                f"production cache missing {len(missing)} assignment entries; "
                f"sample={missing[:8]}"
            )
            if mode == "require":
                raise RuntimeError(msg)
            if progress:
                print(f"{log_prefix} WARNING: {msg}", flush=True)
        return stats

    def prefetch_assignment(
        self,
        assignment: Mapping[str, str],
        *,
        max_resident_bytes: int | None = None,
        max_workers: int = 4,
        require: bool = False,
        progress: bool = False,
        log_prefix: str = "[prod-cache]",
    ) -> dict[str, object]:
        """Prefetch rendered weights required by a concrete assignment.

        ``require`` converts missing entries or resident-budget overflow into
        a hard failure.  That is the production-safe mode for GPU-bound
        recache/export runs because it prevents accidental NVMe streaming.
        """
        keys, missing = self.assignment_keys(assignment)
        nbytes = self.estimate_nbytes(keys)
        budget = (
            int(max_resident_bytes)
            if max_resident_bytes is not None and int(max_resident_bytes) > 0
            else None
        )
        stats: dict[str, object] = {
            "keys": len(keys),
            "missing": len(missing),
            "bytes": int(nbytes),
            "budget_bytes": int(budget or 0),
            "loaded": 0,
            "skipped": False,
        }
        if missing:
            stats["missing_sample"] = missing[:8]
            msg = (
                f"production cache missing {len(missing)} assignment entries; "
                f"sample={missing[:8]}"
            )
            if require:
                raise RuntimeError(msg)
            if progress:
                print(f"{log_prefix} WARNING: {msg}", flush=True)
        if budget is not None and nbytes > budget:
            stats["skipped"] = True
            msg = (
                "production cache preload would exceed resident budget: "
                f"{nbytes / 1024**3:.2f} GiB needed, "
                f"{budget / 1024**3:.2f} GiB budget"
            )
            if require:
                raise RuntimeError(msg)
            if progress:
                print(f"{log_prefix} WARNING: {msg}; skipping preload", flush=True)
            return stats

        if progress:
            print(
                f"{log_prefix} preloading production cache: "
                f"{len(keys)} entries, {nbytes / 1024**3:.2f} GiB",
                flush=True,
            )
        loaded = self.prefetch(keys, max_workers=max_workers)
        stats["loaded"] = int(loaded)
        if progress:
            print(
                f"{log_prefix} preloaded {loaded}/{len(keys)} production "
                "cache entries",
                flush=True,
            )
        return stats

    def _record_lru_load(
        self,
        key: tuple[str, str],
        original_value: object,
        tensor: torch.Tensor,
    ) -> None:
        if self._lru_paths is None:
            self._lru_paths = {}
        if key not in self._lru_paths:
            self._lru_paths[key] = str(original_value)
        if self._lru_order is None:
            return
        if key in self._lru_order:
            self._lru_order.remove(key)
        self._lru_bytes += tensor.element_size() * tensor.numel()
        self._lru_order.append(key)
        self._evict_to_budget()

    def prefetch(self, keys: Sequence[tuple[str, str]] | None = None,
                 max_workers: int = 4) -> int:
        """Eagerly load (a subset of) cache entries via a thread pool.

        ``keys=None`` prefetches every entry that's still on disk (the
        common case at polish startup).  Returns the number of newly-
        materialized tensors.

        Disk-streamed caches typically have torch.load latency ~50 ms
        per file (deserialization-bound, not I/O-bound).  Loading
        serially through 496 entries = ~25 sec; with 4 threads this
        drops to ~6 sec.  Subsequent ``.get()`` calls hit the in-memory
        copy (no torch.load), so per-trial materialization in polish
        becomes essentially free.
        """
        from concurrent.futures import ThreadPoolExecutor

        if keys is None:
            keys = [k for k, v in self.weights.items()
                    if not isinstance(v, torch.Tensor)]
        else:
            keys = [k for k in keys
                    if not isinstance(self.weights.get(k), torch.Tensor)]
        if not keys:
            return 0

        def _load_one(key):
            value = self.weights.get(key)
            if value is None or isinstance(value, torch.Tensor):
                return None
            return (
                key,
                value,
                torch.load(
                    self._path_for_value(value),
                    map_location="cpu",
                    weights_only=True,
                ),
            )

        loaded_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for item in pool.map(_load_one, keys):
                if item is None:
                    continue
                key, original_value, tensor = item
                if isinstance(self.weights.get(key), torch.Tensor):
                    continue
                self.weights[key] = tensor
                self._record_lru_load(key, original_value, tensor)
                loaded_count += 1
        return loaded_count

    def _resolve_to_tensor(self, key: tuple[str, str]) -> torch.Tensor | None:
        """Return the tensor at ``key`` (lazy-load from disk if needed).
        With LRU enabled, the freshly-loaded tensor is bookkept and the
        oldest entries get evicted back to filenames when the byte budget
        is exceeded.  Returns None if the key isn't present."""
        v = self.weights.get(key)
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            # Refresh LRU position.
            if self._lru_order is not None:
                if key in self._lru_order:
                    self._lru_order.remove(key)
                self._lru_order.append(key)
            return v
        # Treat anything non-tensor as a filename / path.
        path = self._path_for_value(v)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        self.weights[key] = loaded
        self._record_lru_load(key, v, loaded)
        return loaded

    def get(self, name: str, fmt: str) -> torch.Tensor | None:
        key = self.resolve_key(name, fmt)
        if key is not None:
            if _is_cb_format_name(key[1]):
                self.validate_cb_render_identity(
                    expected_qnames=[key[0]],
                    require_for_formats=[key[1]],
                    where=f"ProductionWeightCache get({key[0]}@{key[1]})",
                )
            return self._resolve_to_tensor(key)
        return None

    def relocate(self, new_cache_dir: str | Path) -> None:
        """Point the cache at a new on-disk directory of .pt shards.

        Used when a pickled cache is moved to a new host or when a
        cache_dir set inside one container is re-mounted at a different
        path on a second container.  No tensor reload happens here; the
        next ``get()`` will resolve against the new path.
        """
        self.cache_dir = str(new_cache_dir) if new_cache_dir is not None else None

    def verify_files(
        self,
        expected: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, list[tuple[str, str]]]:
        """Verify every disk-resident cache entry's .pt file exists.

        Returns ``{"present": [...], "missing": [...], "in_memory": [...]}``
        keyed by (qname, fmt).  In-memory entries (already-loaded tensors)
        are reported separately and never count as missing.

        On a disk-streaming cache that has been moved or whose backing
        directory was deleted, this is the canonical way to detect the
        problem at startup rather than at first ``get()`` (which raises
        FileNotFoundError mid-polish).  Callers should treat any
        ``missing`` entry as fatal: the cache must be rebuilt or its
        directory restored before use.

        ``expected``, when given, restricts the check to that subset of
        keys.  Default checks every entry in ``self.weights``.
        """
        present: list[tuple[str, str]] = []
        missing: list[tuple[str, str]] = []
        in_memory: list[tuple[str, str]] = []
        keys = list(self.weights) if expected is None else list(expected)
        for key in keys:
            v = self.weights.get(key)
            if v is None:
                missing.append(key)
                continue
            if isinstance(v, torch.Tensor):
                in_memory.append(key)
                continue
            path = str(v)
            if self.cache_dir and not Path(path).is_absolute():
                path = str(Path(self.cache_dir) / path)
            if Path(path).is_file():
                present.append(key)
            else:
                missing.append(key)
        return {"present": present, "missing": missing, "in_memory": in_memory}

    def __contains__(self, key: tuple[str, str]) -> bool:
        # Mirror the alias-resolution that ``get`` performs.
        name, fmt = key
        return self.resolve_key(name, fmt) is not None

    def __len__(self) -> int:
        return len(self.weights)

    def coverage_report(
        self,
        expected_qnames: Sequence[str],
        formats: Sequence[str],
    ) -> dict:
        """Return a dict with ``hits``, ``misses``, ``failed`` lists keyed
        by (qname, fmt).  Use ``validate_coverage`` to raise on any miss.

        Crucially, this checks key membership only — does NOT lazy-load
        tensors — so it stays cheap on disk-streaming caches with
        thousands of entries totalling tens of GB.
        """
        hits: list[tuple[str, str]] = []
        misses: list[tuple[str, str]] = []
        for q in expected_qnames:
            for f in formats:
                if f.upper() == "BF16":
                    continue
                if self.resolve_key(q, f.upper()) is not None:
                    hits.append((q, f.upper()))
                else:
                    misses.append((q, f.upper()))
        return {
            "hits": hits,
            "misses": misses,
            "failed": list((self.failed or {}).keys()),
        }

    def validate_coverage(
        self,
        expected_qnames: Sequence[str],
        formats: Sequence[str],
    ) -> None:
        """Raise ``RuntimeError`` if any (qname, fmt) is missing from the
        cache.  Call this immediately after fill to catch silent gaps
        from naming aliases or render failures."""
        report = self.coverage_report(expected_qnames, formats)
        if report["misses"] or report["failed"]:
            samples = (report["misses"][:5] + report["failed"][:5])
            raise RuntimeError(
                f"ProductionWeightCache coverage failure: "
                f"{len(report['misses'])} misses, "
                f"{len(report['failed'])} failed renders; "
                f"sample={samples}"
            )

class _LinearActivationCollector:
    """Hook every quantizable nn.Linear's input on a forward pass.

    Stores up to ``max_rows`` rows of activations per Linear (concatenated
    across calibration samples) on the configured resident device.  Only handles
    ``nn.Linear`` for now — packed MoE experts route through different
    APIs in the export pipeline and would need a separate collector.

    ``store_qnames`` controls which Linears get full activation tensors
    stored (memory-bounded by ``max_rows``).  All Linears in
    ``qnames`` get a per-Linear scalar ``max_abs`` recorded — that's
    cheap (one float per Linear) and needed by the cache's act-clip
    metadata even for Linears whose render is skipped via resume.
    """

    def __init__(
        self,
        model: nn.Module,
        qnames: set[str],
        max_rows: int,
        store_qnames: set[str] | None = None,
        *,
        store_device: torch.device | str | None = None,
        store_dtype: torch.dtype = torch.float32,
        profile=None,
    ):
        self.model = model
        self.profile = profile
        self.qnames = qnames
        self.store_qnames = set(store_qnames) if store_qnames is not None else set(qnames)
        self.max_rows = int(max_rows)
        self.store_device = torch.device(store_device or "cpu")
        self.store_dtype = store_dtype
        self.activations: dict[str, list[torch.Tensor]] = {}
        self._activation_priorities: dict[str, torch.Tensor] = {}
        self._activation_generator = torch.Generator(device="cpu")
        self._activation_generator.manual_seed(42)
        self.max_abs: dict[str, float] = {}
        self._max_abs_tensors: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._name_by_id: dict[int, str] = {}
        for full_name, mod, attr in iter_quantizable_tensors(model, self.profile):
            if attr != "weight" or not isinstance(mod, nn.Linear):
                continue
            qname = full_name[:-7] if full_name.endswith(".weight") else full_name
            if qname not in qnames and full_name not in qnames:
                continue
            key = qname
            self._name_by_id[id(mod)] = key
            if key in self.store_qnames:
                self.activations[key] = []

    def install(self) -> None:
        for mod_id, key in self._name_by_id.items():
            for full_name, mod, attr in iter_quantizable_tensors(
                self.model,
                self.profile,
            ):
                if id(mod) != mod_id or attr != "weight":
                    continue
                self._handles.append(
                    mod.register_forward_pre_hook(self._make_hook(key))
                )
                break

    def _make_hook(self, key: str):
        def hook(module, args):
            if not args:
                return
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            # Always update the cheap per-Linear max_abs scalar — needed
            # even for Linears we won't store activations for (so cache
            # has act-clip values for every assigned Linear).
            x_abs_max = x.detach().abs().amax()
            prev = self._max_abs_tensors.get(key)
            self._max_abs_tensors[key] = (
                x_abs_max.detach()
                if prev is None
                else torch.maximum(prev, x_abs_max.detach())
            )
            # Only store the full activation tensor if this Linear is in
            # the store set.  Memory bound: store_qnames × max_rows × in.
            if key not in self.store_qnames:
                return
            # NOT non_blocking: async D2H into pageable memory can read
            # the tensor before the producing kernel finishes under GPU
            # contention, deterministically corrupting the snapshot to NaN
            # (2026-07-11, inline-export agent repro).
            flat = x.detach().reshape(-1, x.shape[-1]).to(
                device=self.store_device,
                dtype=self.store_dtype,
            )
            current = (
                torch.cat(self.activations[key], dim=0)
                if self.activations[key]
                else None
            )
            sampled, priorities = update_priority_reservoir(
                current,
                self._activation_priorities.get(key),
                flat,
                max_rows=self.max_rows,
                generator=self._activation_generator,
            )
            self.activations[key] = [] if sampled is None else [sampled]
            if priorities is None:
                self._activation_priorities.pop(key, None)
            else:
                self._activation_priorities[key] = priorities
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def collected(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, parts in self.activations.items():
            if not parts:
                continue
            out[key] = torch.cat(parts, dim=0)
        self.max_abs = {
            key: float(value.detach().to("cpu").item())
            for key, value in self._max_abs_tensors.items()
        }
        return out


@contextmanager
def _temporarily_install_act_aware(
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, object],
):
    """Install module-level state expected by ``_quantize_2d``.

    The export module reads ``_CACHED_ACTIVATIONS`` and ``_ACT_AWARE_FLAGS``
    from its own globals to decide what passes to run.  We mutate these
    inside a try/finally so concurrent export work isn't disturbed.
    """
    from prismaquant import export_native_compressed as enc

    prev_cache = enc._CACHED_ACTIVATIONS
    prev_flags = dict(enc._ACT_AWARE_FLAGS)
    prev_scale_rule = enc._NVFP4_SCALE_RULE
    enc._CACHED_ACTIVATIONS = _DictActivations(activations)
    enc._ACT_AWARE_FLAGS = {
        "gptq": bool(levers.get("gptq", True)),
        "scale_sweep": bool(levers.get("scale_sweep", False)),
        "static_act_order": bool(levers.get("static_act_order", False)),
        "joint_scale_opt": bool(levers.get("joint_scale_opt", False)),
    }
    enc._NVFP4_SCALE_RULE = enc.resolve_nvfp4_scale_rule(
        str(levers.get("nvfp4_scale_rule", "static_6"))
    )
    try:
        yield
    finally:
        enc._CACHED_ACTIVATIONS = prev_cache
        enc._ACT_AWARE_FLAGS.clear()
        enc._ACT_AWARE_FLAGS.update(prev_flags)
        enc._NVFP4_SCALE_RULE = prev_scale_rule


class _DictActivations:
    """`.get(name)` shim matching `_LazyActivationCache`'s interface."""

    def __init__(self, mapping: Mapping[str, torch.Tensor]):
        self._mapping = mapping

    def get(self, name: str) -> torch.Tensor | None:
        a = self._mapping.get(name)
        if a is None and name.endswith(".weight"):
            a = self._mapping.get(name[:-7])
        return a


class _FisherRowWeightCache:
    """Lazy loader for h-detail `g2_per_token` vectors."""

    _FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")

    def __init__(
        self,
        h_detail_dir: str | Path | None,
        fused_sibling_mapping: Mapping[str, Sequence[str]] | None = None,
    ):
        self.detail_dir = Path(h_detail_dir) if h_detail_dir else None
        self.fused_sibling_mapping = {
            str(fused): tuple(str(member) for member in members)
            for fused, members in (fused_sibling_mapping or {}).items()
        }
        self._cache: dict[str, torch.Tensor | None] = {}
        self.loads = 0
        self.misses = 0

    def _path_for_name(self, name: str) -> Path | None:
        if self.detail_dir is None:
            return None
        return self.detail_dir / (self._FNAME_SUB.sub("__", name) + ".pt")

    def _load_exact(self, name: str) -> torch.Tensor | None:
        if self.detail_dir is None:
            return None
        path = self._path_for_name(name)
        if path is None:
            return None
        if not path.is_file():
            return None
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            weights = blob.get("g2_per_token") if isinstance(blob, dict) else None
            if not isinstance(weights, torch.Tensor) or weights.numel() == 0:
                weights = None
            else:
                weights = weights.detach().to(torch.float32).cpu()
        except Exception:
            weights = None
        return weights

    def _split_fused_names(self, qname: str) -> tuple[str, ...]:
        if "." not in qname:
            return ()
        prefix, leaf = qname.rsplit(".", 1)
        members = self.fused_sibling_mapping.get(leaf)
        if not members:
            return ()
        return tuple(f"{prefix}.{member}" for member in members)

    @staticmethod
    def _combine_split_weights(parts: Sequence[torch.Tensor]) -> torch.Tensor | None:
        tensors = [
            p.detach().reshape(-1).to(torch.float32).cpu()
            for p in parts
            if isinstance(p, torch.Tensor) and p.numel() > 0
        ]
        if not tensors:
            return None
        n = min(int(t.numel()) for t in tensors)
        if n <= 0:
            return None
        stacked = torch.stack([t[:n] for t in tensors], dim=0)
        return stacked.mean(dim=0)

    def get(self, qname: str) -> torch.Tensor | None:
        if self.detail_dir is None:
            return None
        if qname in self._cache:
            return self._cache[qname]

        weights = self._load_exact(qname)
        if weights is None:
            split = self._split_fused_names(qname)
            if split:
                parts = [
                    part for name in split
                    if (part := self._load_exact(name)) is not None
                ]
                weights = self._combine_split_weights(parts)

        if weights is None:
            self.misses += 1
        else:
            self.loads += 1
        self._cache[qname] = weights
        return weights


def _fused_sibling_leaf_mapping_from_profile(profile) -> dict[str, tuple[str, ...]]:
    if profile is None:
        return {}
    getter = getattr(profile, "fused_sibling_leaf_mapping", None)
    if not callable(getter):
        return {}
    try:
        mapping = getter()
    except Exception:
        return {}
    return {
        str(fused): tuple(str(member) for member in members)
        for fused, members in (mapping or {}).items()
    }


def _damp_sweep_enabled() -> bool:
    # lazy import: production_weight_cache <-> export_native_compressed
    # are mutually lazy to avoid import cycles
    from prismaquant.export_native_compressed import gptq_damp_sweep_enabled
    return gptq_damp_sweep_enabled()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = int(default)
    return max(lo, min(hi, value))


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = float(default)
    return max(lo, min(hi, value))


@contextmanager
def _temporary_nvfp4_scale_rule(rule: str):
    from prismaquant import export_native_compressed as enc

    previous = enc._NVFP4_SCALE_RULE
    enc._NVFP4_SCALE_RULE = enc.resolve_nvfp4_scale_rule(rule)
    try:
        yield
    finally:
        enc._NVFP4_SCALE_RULE = previous

def _store_rendered_weight_entry(
    *,
    weights: dict[tuple[str, str], object],
    cache_dir_path: Path | None,
    qname: str,
    fmt: str,
    tensor: torch.Tensor,
    weight_dtype: torch.dtype,
) -> None:
    from prismaquant import format_registry as fr

    fmt = fr.canonical_format_name(str(fmt).strip().upper())
    target_dtype = weight_dtype if weight_dtype != torch.float32 else torch.bfloat16
    stored = tensor.to(target_dtype).cpu()
    if cache_dir_path is not None:
        fname = _cache_weight_filename(qname, fmt)
        final_path = cache_dir_path / fname
        tmp_path = cache_dir_path / (fname + ".tmp")
        torch.save(stored, tmp_path)
        os.replace(tmp_path, final_path)
        weights[(qname, fmt)] = fname
        del stored
    else:
        weights[(qname, fmt)] = stored

@dataclass
class _RenderedCandidate:
    label: str
    weight: torch.Tensor
    score: float
    metric: str
    scale_rule: str
    package: tuple[str, ...]
    has_gptq: bool


def _render_score_for_gate(
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor | None,
) -> tuple[float, str]:
    """Score a local render candidate with the shared scorer.

    Activations should normally be present in production cache renders.  The
    weight-MSE fallback keeps pure RTN/FourOverSix unit tests and non-act-aware
    formats measurable without adding a second scoring abstraction.

    Progressive-gate callers must pass the SAME clipped activation matrix the
    GPTQ loop optimized under (``_gate_activation_matrix``), not the raw
    capture: mixing clipped optimization with unclipped gates biases
    accept-vs-RTN decisions near ties by the outlier rows — the exact
    mismatch export's shared-matrix contract
    (``_activation_matrix_for_gptq``) exists to prevent.
    """
    if (
        activations is not None
        and activations.numel() > 0
        and int(activations.shape[-1]) == int(reference_weight.shape[1])
    ):
        return (
            score_render_error(
                reference_weight,
                rendered_weight,
                activations,
                row_weights=None,
            ),
            "output_mse",
        )
    diff = (
        reference_weight.detach().to(torch.float32)
        - rendered_weight.detach().to(
            device=reference_weight.device,
            dtype=torch.float32,
        )
    )
    return float(diff.pow(2).mean().item()), "weight_mse"


def _gate_activation_matrix(
    acts_for_render: torch.Tensor | None,
    cols: int,
    *,
    device: torch.device,
    act_clip_threshold: float | None,
    act_clip_rescale: str | None,
    fisher_row_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return the clip-consistent activation matrix for progressive gates.

    Gates must score candidates under the objective GPTQ optimized: export's
    ``_activation_matrix_for_gptq`` applies the quantile/threshold clip and
    the Fisher row weighting before the Hessian build ("intentionally shared
    by the Hessian build, damping sweep evaluator, and do-no-harm gate"), so
    the cache-fill gate reuses it with identical settings instead of scoring
    on the raw capture.
    """
    if acts_for_render is None:
        return None
    from prismaquant import export_native_compressed as enc

    return enc._activation_matrix_for_gptq(
        acts_for_render,
        int(cols),
        device=device,
        clip_threshold=act_clip_threshold,
        clip_rescale=act_clip_rescale,
        row_weights=fisher_row_weights,
    )


def _render_score_record_key(qname: str, fmt: str) -> str:
    return f"{qname}|{fmt.upper()}"


def _render_score_normalizer(
    reference_weight: torch.Tensor,
    activations: torch.Tensor | None,
    metric: str,
) -> tuple[float, int]:
    rows, cols = reference_weight.shape
    if (
        metric in {"output_mse", "fisher_output_mse"}
        and activations is not None
        and activations.numel() > 0
        and int(activations.shape[-1]) == int(cols)
    ):
        n_act_rows = int(activations.reshape(-1, cols).shape[0])
        return float(max(1, n_act_rows) * int(rows)), n_act_rows
    return float(max(1, int(rows) * int(cols))), 0


def _render_score_record(
    *,
    qname: str,
    fmt: str,
    render_format: str,
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor | None,
    activation_max_abs: float | None,
) -> dict[str, object]:
    raw_score, raw_metric = _render_score_for_gate(
        reference_weight.detach().to(torch.float32),
        rendered_weight,
        activations,
    )
    score = raw_score
    metric = raw_metric
    activation_quantized = False
    activation_clipped = False
    activation_clip_max = (
        activation_max_abs
        if _format_uses_static_activation_clip(fmt) else
        None
    )
    if (
        activations is not None
        and activations.numel() > 0
        and int(activations.shape[-1]) == int(reference_weight.shape[1])
    ):
        try:
            from prismaquant import format_registry as fr

            spec = fr.get_format(fr.canonical_format_name(fmt))
            score, metric, activation_quantized, activation_clipped = (
                _local_forward_render_score(
                    reference_weight=reference_weight,
                    rendered_weight=rendered_weight,
                    activations=activations,
                    activation_quantize=spec.activation_quantize_dequantize,
                    activation_max_abs=activation_clip_max,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"activation-aware render scoring failed for {qname} @ {fmt}: "
                f"{exc}"
            ) from exc
    normalizer, activation_rows = _render_score_normalizer(
        reference_weight,
        activations,
        metric,
    )
    # weight_mse is the original prismaquant cost surrogate: pure
    # (W_orig - W_rendered)^2 averaged over weights. Activation-independent,
    # low variance; the allocator multiplies by h_trace for predicted_dloss.
    ref_f = reference_weight.detach().to(
        device=rendered_weight.device, dtype=torch.float32,
    )
    rendered_f = rendered_weight.detach().to(torch.float32)
    diff = ref_f - rendered_f
    n_weights = int(diff.numel())
    weight_mse = float(diff.pow(2).mean().item()) if n_weights > 0 else 0.0
    rows, cols = reference_weight.shape
    return {
        "qname": str(qname),
        "format": str(fmt).upper(),
        "render_format": str(render_format).upper(),
        "metric": str(metric),
        "score": float(score),
        "score_sum": float(score) * float(normalizer),
        "raw_render_metric": str(raw_metric),
        "raw_render_score": float(raw_score),
        "raw_render_score_sum": float(raw_score) * float(normalizer),
        "weight_mse": float(weight_mse),
        "weight_mse_sum": float(weight_mse) * float(n_weights),
        "n_weights": int(n_weights),
        "normalizer": float(normalizer),
        "activation_rows": int(activation_rows),
        "activation_quantized": bool(activation_quantized),
        "activation_clipped": bool(activation_clipped),
        "activation_max_abs": (
            float(activation_clip_max)
            if activation_clip_max is not None and activation_clip_max > 0
            else None
        ),
        "out_features": int(rows),
        "in_features": int(cols),
    }


def _format_uses_static_activation_clip(fmt: str) -> bool:
    """Return whether local scoring should apply a calibrated activation max.

    NVFP4 serving uses a calibrated tensor-level activation scale plus local
    tensor-group quantization. MXFP8/FP8 dynamic serving computes activation
    scales at runtime, so applying the NVFP4 activation max to those formats
    prices the wrong kernel contract.
    """
    from prismaquant import format_registry as fr

    return fr.canonical_format_name(str(fmt).strip().upper()) == "NVFP4"


def _local_forward_render_score(
    *,
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor,
    activation_quantize,
    activation_max_abs: float | None,
    row_chunk: int = 128,
) -> tuple[float, str, bool, bool]:
    rows, cols = reference_weight.shape
    if rendered_weight.shape != reference_weight.shape:
        return float("inf"), "output_mse", False, False
    if activations.shape[-1] != cols:
        return float("inf"), "output_mse", False, False
    device = reference_weight.device
    ref_t = reference_weight.detach().to(device=device, dtype=torch.float32).t()
    rendered_t = rendered_weight.detach().to(device=device, dtype=torch.float32).t()
    x = activations.detach().to(device=device, dtype=torch.float32).reshape(-1, cols)
    clipped = False
    if (
        activation_max_abs is not None
        and float(activation_max_abs) > 0
        and _env_flag("PRISMAQUANT_PROD_ACT_SCALES", True)
    ):
        x_quant_input = x.clamp(-float(activation_max_abs), float(activation_max_abs))
        clipped = True
    else:
        x_quant_input = x

    total = torch.zeros((), dtype=torch.float32, device=device)
    quantized_any = False
    with torch.no_grad():
        for start in range(0, x.shape[0], int(row_chunk)):
            x_ref = x[start:start + int(row_chunk)]
            x_q = activation_quantize(x_quant_input[start:start + int(row_chunk)])
            if x_q is not x_ref:
                quantized_any = quantized_any or not torch.equal(
                    x_q.detach().to(device=device, dtype=torch.float32),
                    x_ref,
                )
            x_q = x_q.to(device=device, dtype=torch.float32)
            y_ref = x_ref @ ref_t
            y_q = x_q @ rendered_t
            err = (y_ref - y_q).pow(2)
            total = total + err.sum()
    score = float(total.item()) / max(1, int(x.shape[0]) * int(rows))
    return score, "output_mse", bool(quantized_any), bool(clipped)


def _load_render_score_sidecar(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.is_file():
        return {}
    import json as _json

    try:
        raw = _json.loads(path.read_text())
    except Exception:
        return {}
    records = raw.get("records") if isinstance(raw, Mapping) else None
    if not isinstance(records, Mapping):
        return {}
    out: dict[str, dict[str, object]] = {}
    for key, value in records.items():
        if isinstance(value, Mapping):
            out[str(key)] = dict(value)
    return out


def _write_render_score_sidecar(
    path: Path | None,
    records: Mapping[str, Mapping[str, object]],
) -> None:
    if path is None:
        return
    import json as _json

    payload = {
        "schema": "prismaquant.production_render_scores.v1",
        "records": dict(sorted((str(k), dict(v)) for k, v in records.items())),
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _format_supports_render_mechanism(fmt: str, mechanism: str) -> bool:
    """Return whether a shared render mechanism is meaningful for ``fmt``.

    The production render pipeline is format-agnostic in order. Individual
    formats opt out of mechanisms whose math or exported schema does not
    apply.
    """

    fmt_u = str(fmt).strip().upper()
    mech = str(mechanism).strip()
    if fmt_u == "NVFP4":
        return mech in {
            "four_over_six",
            "gptq",
            "static_act_order",
            "joint_scale_opt",
            "fisher_gptq",
            "scale_sweep",
        }
    if fmt_u in {"FP8_E4M3", "FP8_E5M2"}:
        return mech == "gptq" or (mech == "scale_sweep" and fmt_u == "FP8_E4M3")
    if fmt_u == "MXFP4":
        return mech in {"gptq", "static_act_order"}
    if fmt_u in {"MXFP8_E4M3", "MXFP8_E5M2"}:
        return mech in {"gptq", "static_act_order"} or (
            mech == "scale_sweep" and fmt_u == "MXFP8_E4M3"
        )
    if _weighted_render_family(fmt_u) is not None:
        # VQ codebooks quantize 8-column vectors jointly and k-quant
        # superblocks pick scales per block; the scalar-column mechanisms
        # (gptq/jso/scale_sweep/act_order) do not apply to either. The
        # imatrix-weighted search IS the deliberate render (col_weights) —
        # re-vet R3 / CB Milestone C made it reachable from this path.
        return mech == "weighted_vq"
    return False


# Format families whose EXPORTER renders imatrix-weighted, so a production
# render of the same (qname, fmt) is only exporter-faithful when it applies the
# same per-input-column vector. Keyed off the family exactly as
# `measure_quant_cost._cost_render_uses_imatrix` keys the inline cost render —
# one invariant, two call sites, no third opinion.
WEIGHTED_RENDER_FAMILIES = ("nvfp4_cb", "fp8_cb", "gguf")


def _expert_col_weights(
    stack_cw: torch.Tensor | None, index: int, n_experts: int,
) -> torch.Tensor | None:
    """Slice a packed-expert imatrix entry for expert ``index``.

    **Delegates** to ``measure_quant_cost._item_col_weights`` rather than
    re-deriving the rule: a ``(E, in)`` stack is indexed per expert, anything
    else is a single pooled vector shared by every expert. One definition, two
    readers — if the cost stage and the cache ever sliced the same pickle
    differently they would weight the same tensor differently, which is the
    rendering confound wearing a new hat.
    """
    if stack_cw is None:
        return None
    from prismaquant.measure_quant_cost import _item_col_weights

    return _item_col_weights(stack_cw, index, n_experts)


def _weighted_render_family(fmt: str) -> str | None:
    """The weighted-render family of ``fmt``, or ``None`` for every format
    whose render ignores ``col_weights`` (NVFP4/FP8/MX/BF16/INT)."""
    from prismaquant import format_registry as fr

    try:
        family = fr.get_format(str(fmt).strip().upper()).family
    except Exception:
        return None
    return family if family in WEIGHTED_RENDER_FAMILIES else None


def _is_cb_format_name(fmt: str) -> bool:
    """Return whether ``fmt`` is one of the two serialized CB families."""
    from prismaquant import format_registry as fr

    try:
        return fr.get_format(
            fr.canonical_format_name(str(fmt).strip().upper())
        ).family in {"nvfp4_cb", "fp8_cb"}
    except Exception:
        return False


def _cb_qnames_in_render_scope(
    formats_by_qname: Mapping[str, str | Sequence[str]],
) -> list[str]:
    return list(_canonical_cb_render_scope(formats_by_qname))


def _canonical_cb_render_scope(
    formats_by_qname: Mapping[str, str | Sequence[str]],
) -> dict[str, list[str]]:
    """Canonical qname -> serialized-CB-format render scope."""
    from prismaquant import format_registry as fr

    scope: dict[str, list[str]] = {}
    for raw_qname, raw_formats in formats_by_qname.items():
        formats = (
            (raw_formats,)
            if isinstance(raw_formats, str)
            else tuple(raw_formats)
        )
        canonical = sorted({
            fr.canonical_format_name(str(fmt).strip().upper())
            for fmt in formats
            if _is_cb_format_name(fmt)
        })
        if canonical:
            scope[str(raw_qname)] = canonical
    return dict(sorted(scope.items()))


def _canonical_cb_col_weights_identity(
    col_weights: Mapping[str, torch.Tensor] | None,
    qnames: Sequence[str],
) -> tuple[str, dict[str, list[int]], dict[str, str]]:
    """Hash the exact imatrix values that make a CB render reproducible.

    The byte stream is version-tagged and consists of sorted, length-framed
    ``qname``, tensor ``shape``, and contiguous little-endian CPU float32
    values.  Length framing prevents concatenation ambiguity, and the explicit
    dtype/endian conversion makes insertion order, source dtype, device, and
    striding irrelevant to the identity.
    """
    if not isinstance(col_weights, Mapping):
        raise ValueError(
            "production CB render requires an explicit col_weights mapping "
            "for its persisted render identity"
        )

    names = sorted(dict.fromkeys(str(name) for name in qnames))
    missing = [name for name in names if name not in col_weights]
    if missing:
        raise ValueError(
            "production CB render identity is missing col_weights for "
            f"{len(missing)} qname(s); sample={missing[:8]}"
        )

    digest = hashlib.sha256()
    digest.update((CB_COL_WEIGHTS_HASH_SCHEMA + "\0").encode("utf-8"))
    shapes: dict[str, list[int]] = {}
    content_digests: dict[str, str] = {}

    def _frame(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
        digest.update(payload)

    for qname in names:
        tensor = torch.as_tensor(col_weights[qname]).detach().to(
            device="cpu",
            dtype=torch.float32,
        ).contiguous()
        if tensor.layout != torch.strided:
            raise ValueError(
                f"production CB col_weights for {qname!r} must be strided"
            )
        shape = [int(dim) for dim in tensor.shape]
        shapes[qname] = shape
        _frame(qname.encode("utf-8"))
        _frame(json.dumps(shape, separators=(",", ":")).encode("ascii"))
        # ``<f4`` fixes little-endian float32 even on a big-endian producer.
        raw = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        content_digests[qname] = hashlib.sha256(raw).hexdigest()
        _frame(raw)
    return digest.hexdigest(), shapes, content_digests


def canonical_cb_col_weights_sha256(
    col_weights: Mapping[str, torch.Tensor] | None,
    qnames: Sequence[str],
) -> str:
    """Public digest helper for producer provenance and focused audits."""
    digest, _, _ = _canonical_cb_col_weights_identity(col_weights, qnames)
    return digest


def _canonical_json_value(value, *, where: str):
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not canonical JSON data") from exc
    return json.loads(encoded)


def _cb_render_contract(
    levers: Mapping[str, object] | None,
    mechanism_plan,
    cb_formats: Sequence[str],
) -> dict[str, object]:
    if not isinstance(levers, Mapping):
        raise ValueError(
            "production CB render identity requires explicit resolved render "
            "levers"
        )
    if not bool(levers.get("weighted_vq", True)):
        raise ValueError(
            "production CB render identity requires weighted_vq=True"
        )
    ordered = getattr(mechanism_plan, "ordered", mechanism_plan)
    if ordered is None:
        raise ValueError(
            "production CB render identity requires an explicit resolved "
            "render mechanism plan"
        )
    records: list[dict[str, object]] = []
    calibration_dependent: list[str] = []
    for raw in ordered:
        if isinstance(raw, Mapping):
            record = {
                str(key): raw[key]
                for key in (
                    "name", "operation", "scope", "phase", "gate_metric"
                )
                if key in raw
            }
            name = str(record.get("name", ""))
        else:
            name = str(getattr(raw, "name", raw))
            record = {
                "name": name,
                **({
                    key: getattr(raw, key)
                    for key in (
                        "operation", "scope", "phase", "gate_metric"
                    )
                    if hasattr(raw, key)
                }),
            }
        records.append(_canonical_json_value(
            record,
            where="CB render mechanism record",
        ))
        if any(
            _format_supports_render_mechanism(fmt, name)
            and name != "weighted_vq"
            for fmt in cb_formats
        ):
            calibration_dependent.append(name)
    if calibration_dependent:
        raise ValueError(
            "production CB render unexpectedly enables calibration-dependent "
            f"mechanisms {sorted(set(calibration_dependent))}; bind their "
            "inputs before admitting them to the CB contract"
        )
    return {
        "schema": CB_RENDER_CONTRACT_SCHEMA,
        "mechanism_abi": CB_RENDER_MECHANISM_ABI,
        "resolved_levers": _canonical_json_value(
            dict(levers),
            where="CB resolved render levers",
        ),
        "mechanism_plan": records,
        "cb_active_mechanisms": ["weighted_vq"],
        "calibration_dependent_mechanisms": [],
    }


def _source_weight_value_identity(
    weight: torch.Tensor,
) -> tuple[list[int], str]:
    tensor = torch.as_tensor(weight).detach()
    shape = [int(dim) for dim in tensor.shape]
    digest = hashlib.sha256()
    max_chunk_elements = 4 * 1024 * 1024  # <=16 MiB after fp32 conversion

    def _chunks_c_order(value: torch.Tensor):
        if value.numel() <= max_chunk_elements or value.ndim == 0:
            yield value
            return
        trailing = math.prod(int(dim) for dim in value.shape[1:])
        if trailing <= max_chunk_elements:
            step = max(1, max_chunk_elements // max(trailing, 1))
            for start in range(0, int(value.shape[0]), step):
                yield value[start:start + step]
            return
        # A single leading slice is still too large. Recurse dimension by
        # dimension; concatenating these chunks is exactly C-order traversal.
        for index in range(int(value.shape[0])):
            yield from _chunks_c_order(value[index])

    for chunk in _chunks_c_order(tensor):
        cpu = chunk.to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(
            cpu.numpy().astype("<f4", copy=False).tobytes(order="C")
        )
        del cpu
    return shape, digest.hexdigest()


def _combined_source_weights_sha256(
    shapes: Mapping[str, Sequence[int]],
    content_digests: Mapping[str, str],
) -> str:
    digest = hashlib.sha256()
    digest.update((CB_SOURCE_WEIGHTS_HASH_SCHEMA + "\0").encode("utf-8"))

    def _frame(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "little", signed=False))
        digest.update(payload)

    for qname in sorted(content_digests):
        _frame(qname.encode("utf-8"))
        _frame(json.dumps(
            [int(dim) for dim in shapes[qname]],
            separators=(",", ":"),
        ).encode("ascii"))
        _frame(bytes.fromhex(str(content_digests[qname])))
    return digest.hexdigest()


def validate_cb_render_source_weight(
    identity: Mapping[str, object],
    qname: str,
    weight: torch.Tensor,
    *,
    where: str,
) -> None:
    """Incrementally verify one decoded exporter source tensor."""
    if (
        not isinstance(identity, Mapping)
        or identity.get("schema") != CB_RENDER_IDENTITY_SCHEMA
        or identity.get("source_weights_complete") is not True
    ):
        raise ValueError(
            f"{where}: missing source-complete CB render identity"
        )
    name = str(qname)
    shapes = identity.get("source_weights_shapes")
    content = identity.get("source_weights_content_sha256")
    if (
        not isinstance(shapes, Mapping)
        or not isinstance(content, Mapping)
        or name not in shapes
        or name not in content
    ):
        raise ValueError(
            f"{where}: decoded CB source qname {name!r} is absent from the "
            "selected render identity"
        )
    shape, digest = _source_weight_value_identity(weight)
    if (
        shape != [int(dim) for dim in shapes[name]]
        or digest != str(content[name]).lower()
    ):
        raise ValueError(
            f"{where}: decoded CB source-weight value differs for {name!r}"
        )


def build_production_cache_cb_render_identity(
    formats_by_qname: Mapping[str, str | Sequence[str]],
    *,
    cb_serialization_context,
    col_weights: Mapping[str, torch.Tensor] | None,
    render_levers: Mapping[str, object] | None,
    render_mechanism_plan,
) -> dict[str, object] | None:
    """Build the fresh, versioned identity for a CB cache render scope.

    ``None`` is returned only when the scope has no CB formats.  A CB scope
    requires a caller-supplied ``CBSerializationContext``; this function never
    consults process environment defaults.
    """
    scope = _canonical_cb_render_scope(formats_by_qname)
    qnames = list(scope)
    if not scope:
        return None
    if cb_serialization_context is None:
        raise ValueError(
            "production CB render requires an explicit "
            "CBSerializationContext; refusing to infer layout/codebook "
            "identity from the current environment"
        )
    from prismaquant.nvfp4_cb_footprint import (
        CBSerializationContext,
        cb_serialization_context_stamp,
    )

    if not isinstance(cb_serialization_context, CBSerializationContext):
        raise TypeError(
            "cb_serialization_context must be a CBSerializationContext, got "
            f"{type(cb_serialization_context).__name__}"
        )
    col_digest, shapes, content_digests = _canonical_cb_col_weights_identity(
        col_weights,
        qnames,
    )
    cb_formats = sorted({fmt for formats in scope.values() for fmt in formats})
    return {
        "schema": CB_RENDER_IDENTITY_SCHEMA,
        "cb_serialized_payload": cb_serialization_context_stamp(
            cb_serialization_context,
            formats=cb_formats,
        ),
        "render_contract": _cb_render_contract(
            render_levers,
            render_mechanism_plan,
            cb_formats,
        ),
        "cb_formats_by_qname": scope,
        "col_weights_schema": CB_COL_WEIGHTS_HASH_SCHEMA,
        "col_weights_sha256": col_digest,
        "col_weights_entries": len(qnames),
        "col_weights_qnames": qnames,
        "col_weights_shapes": shapes,
        "col_weights_content_sha256": content_digests,
        "source_weights_schema": CB_SOURCE_WEIGHTS_HASH_SCHEMA,
        "source_weights_complete": False,
        "source_weights_sha256": None,
        "source_weights_shapes": {},
        "source_weights_content_sha256": {},
    }


def validate_cb_render_identity_metadata(
    identity: Mapping[str, object] | None,
    *,
    expected_context=None,
    expected_qnames: Sequence[str] | None = None,
    expected_formats_by_qname: Mapping[
        str, str | Sequence[str]
    ] | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    source_weights: Mapping[str, torch.Tensor] | None = None,
    require_source_complete: bool = True,
    require_minchain_cells: bool = False,
    where: str = "CB render identity",
):
    """Validate a value-bearing CB render identity independent of a cache.

    This is the shared consumer gate for cache, cost, assignment, and export
    artifacts.  It validates the serialized layout/codebook bytes, exact
    qname/format scope, and canonical float32 imatrix content without ever
    consulting the current process environment.
    """
    if not isinstance(identity, Mapping):
        raise ValueError(f"{where}: missing versioned CB render identity")
    if identity.get("schema") != CB_RENDER_IDENTITY_SCHEMA:
        raise ValueError(
            f"{where}: unsupported CB render identity schema "
            f"{identity.get('schema')!r}"
        )
    if identity.get("col_weights_schema") != CB_COL_WEIGHTS_HASH_SCHEMA:
        raise ValueError(
            f"{where}: unsupported CB col_weights identity schema "
            f"{identity.get('col_weights_schema')!r}"
        )
    raw_qnames = identity.get("col_weights_qnames")
    if not isinstance(raw_qnames, Sequence) or isinstance(raw_qnames, str):
        raise ValueError(f"{where}: CB render identity has no qname list")
    identity_qnames = [str(name) for name in raw_qnames]
    if identity_qnames != sorted(set(identity_qnames)):
        raise ValueError(
            f"{where}: CB render identity qnames are not canonical/sorted"
        )
    if int(identity.get("col_weights_entries", -1)) != len(identity_qnames):
        raise ValueError(f"{where}: CB render identity entry count differs")

    raw_scope = identity.get("cb_formats_by_qname")
    if not isinstance(raw_scope, Mapping):
        raise ValueError(f"{where}: CB render identity has no format scope")
    try:
        scope = _canonical_cb_render_scope(raw_scope)
    except Exception as exc:
        raise ValueError(
            f"{where}: CB render identity format scope is invalid"
        ) from exc
    normalized_raw_scope = {
        str(name): [str(fmt) for fmt in formats]
        for name, formats in raw_scope.items()
        if isinstance(formats, Sequence) and not isinstance(formats, str)
    }
    if (
        set(scope) != set(identity_qnames)
        or normalized_raw_scope != scope
    ):
        raise ValueError(
            f"{where}: CB render identity format scope is not exact, "
            "canonical, and qname-complete"
        )

    render_contract = identity.get("render_contract")
    if not isinstance(render_contract, Mapping):
        raise ValueError(f"{where}: CB render contract is missing")
    if render_contract.get("schema") != CB_RENDER_CONTRACT_SCHEMA:
        raise ValueError(f"{where}: unsupported CB render contract schema")
    if render_contract.get("mechanism_abi") != CB_RENDER_MECHANISM_ABI:
        raise ValueError(f"{where}: unsupported CB render mechanism ABI")
    if not isinstance(render_contract.get("resolved_levers"), Mapping):
        raise ValueError(f"{where}: CB render contract has no resolved levers")
    if not isinstance(render_contract.get("mechanism_plan"), Sequence):
        raise ValueError(f"{where}: CB render contract has no mechanism plan")
    if render_contract.get("cb_active_mechanisms") != ["weighted_vq"]:
        raise ValueError(f"{where}: unsupported CB active render mechanisms")
    if render_contract.get("calibration_dependent_mechanisms") != []:
        raise ValueError(
            f"{where}: CB render identity contains unbound "
            "calibration-dependent mechanisms"
        )

    raw_shapes = identity.get("col_weights_shapes")
    if not isinstance(raw_shapes, Mapping) or set(raw_shapes) != set(identity_qnames):
        raise ValueError(
            f"{where}: CB render identity shapes do not cover its qnames"
        )
    shapes: dict[str, list[int]] = {}
    for qname in identity_qnames:
        raw_shape = raw_shapes[qname]
        if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, str):
            raise ValueError(
                f"{where}: CB render identity shape for {qname!r} is invalid"
            )
        shape = [int(dim) for dim in raw_shape]
        if any(dim < 0 for dim in shape):
            raise ValueError(
                f"{where}: CB render identity shape for {qname!r} is invalid"
            )
        shapes[qname] = shape
    raw_content_digests = identity.get("col_weights_content_sha256")
    if (
        not isinstance(raw_content_digests, Mapping)
        or set(raw_content_digests) != set(identity_qnames)
    ):
        raise ValueError(
            f"{where}: CB render identity content hashes do not cover its "
            "qnames"
        )
    content_digests = {
        qname: str(raw_content_digests[qname]).lower()
        for qname in identity_qnames
    }
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in content_digests.values()
    ):
        raise ValueError(f"{where}: CB col_weights content SHA-256 is invalid")
    observed_digest = str(identity.get("col_weights_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", observed_digest):
        raise ValueError(f"{where}: CB col_weights SHA-256 is invalid")

    if identity.get("source_weights_schema") != CB_SOURCE_WEIGHTS_HASH_SCHEMA:
        raise ValueError(
            f"{where}: unsupported CB source-weight identity schema"
        )
    source_shapes_raw = identity.get("source_weights_shapes")
    source_content_raw = identity.get("source_weights_content_sha256")
    if not isinstance(source_shapes_raw, Mapping) or not isinstance(
        source_content_raw, Mapping
    ):
        raise ValueError(f"{where}: CB source-weight identity is malformed")
    source_names = set(str(name) for name in source_content_raw)
    if source_names != set(str(name) for name in source_shapes_raw):
        raise ValueError(
            f"{where}: CB source-weight shapes/content scopes differ"
        )
    if not source_names.issubset(identity_qnames):
        raise ValueError(
            f"{where}: CB source-weight identity contains out-of-scope qnames"
        )
    source_shapes: dict[str, list[int]] = {}
    source_content: dict[str, str] = {}
    for qname in sorted(source_names):
        raw_shape = source_shapes_raw[qname]
        if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, str):
            raise ValueError(
                f"{where}: source-weight shape for {qname!r} is invalid"
            )
        source_shapes[qname] = [int(dim) for dim in raw_shape]
        value = str(source_content_raw[qname]).lower()
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(
                f"{where}: source-weight SHA-256 for {qname!r} is invalid"
            )
        source_content[qname] = value
    source_complete = identity.get("source_weights_complete")
    if not isinstance(source_complete, bool):
        raise ValueError(
            f"{where}: CB source-weight completeness marker is invalid"
        )
    if source_complete:
        if source_names != set(identity_qnames):
            raise ValueError(
                f"{where}: complete CB source-weight identity does not cover "
                "its qname scope"
            )
        expected_source_digest = _combined_source_weights_sha256(
            source_shapes,
            source_content,
        )
        if identity.get("source_weights_sha256") != expected_source_digest:
            raise ValueError(
                f"{where}: combined CB source-weight SHA-256 differs"
            )
    elif require_source_complete:
        raise ValueError(
            f"{where}: CB source-weight identity is incomplete; cache render "
            "did not observe every exact decoded source tensor"
        )
    elif identity.get("source_weights_sha256") is not None:
        raise ValueError(
            f"{where}: incomplete CB source identity has a final digest"
        )

    if expected_qnames is not None:
        missing_expected = sorted(
            set(str(name) for name in expected_qnames) - set(identity_qnames)
        )
        if missing_expected:
            raise ValueError(
                f"{where}: requested CB qnames are absent from render "
                f"identity; sample={missing_expected[:8]}"
            )
    if expected_formats_by_qname is not None:
        expected_scope = _canonical_cb_render_scope(expected_formats_by_qname)
        missing_or_mismatched = sorted(
            qname for qname, formats in expected_scope.items()
            if qname not in scope
            or not set(formats).issubset(scope[qname])
        )
        if missing_or_mismatched:
            raise ValueError(
                f"{where}: requested CB qname/format scope differs from the "
                f"stored render; sample={missing_or_mismatched[:8]}"
            )

    from prismaquant.nvfp4_cb_footprint import (
        cb_serialization_context_from_stamp,
        codebook_source_for_format,
        lattice_codebook_content_sha256,
        validate_cb_serialization_context_stamp,
    )

    stamp = identity.get("cb_serialized_payload")
    context = cb_serialization_context_from_stamp(
        stamp,
        where=f"{where} serialized payload",
    )
    if expected_context is not None:
        validate_cb_serialization_context_stamp(
            stamp,
            expected_context,
            where=where,
        )
    formats = sorted({fmt for values in scope.values() for fmt in values})
    lattice_formats = [
        fmt
        for fmt in formats
        if codebook_source_for_format(fmt, context) == "lattice"
    ]
    learned_formats = [
        fmt
        for fmt in formats
        if codebook_source_for_format(fmt, context) == "learned"
    ]
    if lattice_formats:
        observed_lattice = stamp.get("lattice_codebook_sha256_by_format")
        expected_lattice = {
            fmt: list(lattice_codebook_content_sha256(fmt))
            for fmt in lattice_formats
        }
        if not isinstance(observed_lattice, Mapping) or dict(
            observed_lattice
        ) != expected_lattice:
            raise ValueError(
                f"{where}: CB render identity does not bind the exact "
                "canonical lattice codebook bytes for its format scope"
            )
    if learned_formats:
        learned_digests = stamp.get("codebook_content_sha256")
        if not isinstance(learned_digests, Mapping) or not learned_digests:
            raise ValueError(
                f"{where}: learned CB render identity has no value-bearing "
                "codebook content digests"
            )

    raw_minchain_cells = identity.get("cb_minchain_cells")
    if not context.minchain:
        if raw_minchain_cells is not None:
            raise ValueError(
                f"{where}: non-chain render claims per-cell min-chain identity"
            )
    elif raw_minchain_cells is None:
        if require_minchain_cells:
            raise ValueError(
                f"{where}: min-chain render is missing per-cell arm/digest "
                "identity"
            )
    else:
        if not isinstance(raw_minchain_cells, Mapping):
            raise ValueError(f"{where}: min-chain cell identity is malformed")
        from prismaquant.cb_minchain import validate_chain_identity

        normalized_cells: dict[str, dict[str, list[dict[str, object]]]] = {}
        for raw_qname, raw_formats in raw_minchain_cells.items():
            qname = str(raw_qname)
            if qname not in scope or not isinstance(raw_formats, Mapping):
                raise ValueError(
                    f"{where}: min-chain cell identity has an out-of-scope qname"
                )
            normalized_cells[qname] = {}
            for raw_fmt, raw_cells in raw_formats.items():
                fmt = str(raw_fmt)
                if (
                    fmt not in scope[qname]
                    or not isinstance(raw_cells, Sequence)
                    or isinstance(raw_cells, str)
                    or not raw_cells
                ):
                    raise ValueError(
                        f"{where}: min-chain cell identity has an invalid "
                        f"scope at {qname}/{fmt}"
                    )
                normalized_cells[qname][fmt] = [
                    validate_chain_identity(
                        cell,
                        where=f"{where} {qname}/{fmt}[{index}]",
                    )
                    for index, cell in enumerate(raw_cells)
                ]
        if normalized_cells != raw_minchain_cells:
            raise ValueError(
                f"{where}: min-chain cell identity is not canonical"
            )
        if require_minchain_cells:
            missing = sorted(
                f"{qname}/{fmt}"
                for qname, formats in scope.items()
                for fmt in formats
                if fmt not in normalized_cells.get(qname, {})
            )
            if missing:
                raise ValueError(
                    f"{where}: min-chain cell identity does not cover the "
                    f"selected scope; sample={missing[:8]}"
                )

    if col_weights is not None:
        expected_digest, expected_shapes, expected_content = (
            _canonical_cb_col_weights_identity(
                col_weights,
                identity_qnames,
            )
        )
        if (
            expected_digest != observed_digest
            or expected_shapes != shapes
            or expected_content != content_digests
        ):
            raise ValueError(
                f"{where}: CB col_weights identity differs from the stored "
                "render"
            )
    if source_weights is not None:
        missing_source_values = sorted(
            set(identity_qnames) - set(str(name) for name in source_weights)
        )
        if missing_source_values:
            raise ValueError(
                f"{where}: source-weight validation mapping is missing "
                f"qnames; sample={missing_source_values[:8]}"
            )
        mismatched_source_values: list[str] = []
        for qname in identity_qnames:
            shape, digest = _source_weight_value_identity(
                source_weights[qname]
            )
            if (
                shape != source_shapes.get(qname)
                or digest != source_content.get(qname)
            ):
                mismatched_source_values.append(qname)
        if mismatched_source_values:
            raise ValueError(
                f"{where}: decoded CB source-weight values differ; "
                f"sample={mismatched_source_values[:8]}"
            )
    return context


def bind_cb_render_identity_source_weights(
    identity: Mapping[str, object],
    source_weights: Mapping[str, torch.Tensor],
    *,
    require_complete: bool = True,
    where: str = "CB render identity source binding",
) -> dict[str, object]:
    """Bind exact decoded source tensors into a render identity copy."""
    validate_cb_render_identity_metadata(
        identity,
        require_source_complete=False,
        where=where,
    )
    updated = copy.deepcopy(dict(identity))
    scope = set(str(name) for name in updated["col_weights_qnames"])
    shapes = dict(updated["source_weights_shapes"])
    content = dict(updated["source_weights_content_sha256"])
    for raw_qname, weight in source_weights.items():
        qname = str(raw_qname)
        if qname not in scope:
            continue
        shape, digest = _source_weight_value_identity(weight)
        if qname in content and (
            list(shapes[qname]) != shape
            or str(content[qname]).lower() != digest
        ):
            raise ValueError(
                f"{where}: decoded source weight changed for {qname!r}"
            )
        shapes[qname] = shape
        content[qname] = digest
    observed = set(content)
    complete = observed == scope
    if require_complete and not complete:
        missing = sorted(scope - observed)
        raise ValueError(
            f"{where}: missing decoded source weights; sample={missing[:8]}"
        )
    updated["source_weights_shapes"] = dict(sorted(shapes.items()))
    updated["source_weights_content_sha256"] = dict(sorted(content.items()))
    updated["source_weights_complete"] = bool(complete)
    updated["source_weights_sha256"] = (
        _combined_source_weights_sha256(shapes, content)
        if complete else None
    )
    validate_cb_render_identity_metadata(
        updated,
        require_source_complete=require_complete,
        where=where,
    )
    return updated


def bind_production_cache_cb_source_weights(
    cache: ProductionWeightCache,
    source_weights: Mapping[str, torch.Tensor],
    *,
    require_complete: bool = False,
    where: str = "ProductionWeightCache source binding",
) -> None:
    """Record decoded source values as streamed CB tensors become resident."""
    metadata = dict(getattr(cache, "metadata", {}) or {})
    identity = metadata.get(CB_RENDER_IDENTITY_METADATA_KEY)
    if identity is None:
        if source_weights:
            raise ValueError(f"{where}: cache has no CB render identity")
        return
    metadata[CB_RENDER_IDENTITY_METADATA_KEY] = (
        bind_cb_render_identity_source_weights(
            identity,
            source_weights,
            require_complete=require_complete,
            where=where,
        )
    )
    cache.metadata = metadata


def project_cb_render_identity(
    identity: Mapping[str, object],
    formats_by_qname: Mapping[str, str | Sequence[str]],
    *,
    col_weights: Mapping[str, torch.Tensor],
    where: str,
) -> dict[str, object]:
    """Project a full-menu identity onto one exact selected CB assignment."""
    context = validate_cb_render_identity_metadata(identity, where=where)
    selected_scope = _canonical_cb_render_scope(formats_by_qname)
    if not selected_scope:
        raise ValueError(f"{where}: selected scope contains no CB formats")
    original_scope = identity["cb_formats_by_qname"]
    missing = sorted(
        qname for qname, formats in selected_scope.items()
        if qname not in original_scope
        or not set(formats).issubset(original_scope[qname])
    )
    if missing:
        raise ValueError(
            f"{where}: selected CB scope is absent from producer identity; "
            f"sample={missing[:8]}"
        )
    contract = identity["render_contract"]
    projected = build_production_cache_cb_render_identity(
        selected_scope,
        cb_serialization_context=context,
        col_weights=col_weights,
        render_levers=contract["resolved_levers"],
        render_mechanism_plan=contract["mechanism_plan"],
    )
    if projected["render_contract"] != contract:
        raise ValueError(
            f"{where}: selected projection changed the resolved render "
            "contract"
        )
    projected_shapes = {
        qname: list(identity["source_weights_shapes"][qname])
        for qname in selected_scope
    }
    projected_content = {
        qname: str(identity["source_weights_content_sha256"][qname])
        for qname in selected_scope
    }
    projected["source_weights_shapes"] = projected_shapes
    projected["source_weights_content_sha256"] = projected_content
    projected["source_weights_complete"] = True
    projected["source_weights_sha256"] = _combined_source_weights_sha256(
        projected_shapes,
        projected_content,
    )
    if context.minchain and "cb_minchain_cells" in identity:
        original_cells = identity["cb_minchain_cells"]
        projected["cb_minchain_cells"] = {
            qname: {
                fmt: copy.deepcopy(original_cells[qname][fmt])
                for fmt in formats
            }
            for qname, formats in selected_scope.items()
        }
    validate_cb_render_identity_metadata(
        projected,
        col_weights=col_weights,
        where=f"{where} projected identity",
    )
    return projected


def merge_cb_render_identities(
    identities: Sequence[Mapping[str, object]],
    *,
    col_weights: Mapping[str, torch.Tensor],
    where: str,
) -> dict[str, object] | None:
    """Merge disjoint source-complete shard identities into one cost scope."""
    if not identities:
        return None
    contexts = [
        validate_cb_render_identity_metadata(
            identity,
            col_weights=col_weights,
            where=f"{where} shard {index}",
        )
        for index, identity in enumerate(identities)
    ]
    first = identities[0]
    contract = first["render_contract"]
    scope: dict[str, list[str]] = {}
    source_shapes: dict[str, list[int]] = {}
    source_content: dict[str, str] = {}
    minchain_cells: dict[str, dict[str, list[dict[str, object]]]] = {}
    for index, identity in enumerate(identities):
        if identity["render_contract"] != contract:
            raise ValueError(
                f"{where}: CB shard {index} render contract differs"
            )
        from prismaquant.nvfp4_cb_footprint import (
            validate_cb_serialization_context_stamp,
        )

        validate_cb_serialization_context_stamp(
            identity["cb_serialized_payload"],
            contexts[0],
            where=f"{where} shard {index}",
        )
        overlap = set(scope) & set(identity["cb_formats_by_qname"])
        if overlap:
            raise ValueError(
                f"{where}: CB shard render identities overlap; "
                f"sample={sorted(overlap)[:8]}"
            )
        scope.update(copy.deepcopy(identity["cb_formats_by_qname"]))
        source_shapes.update(copy.deepcopy(identity["source_weights_shapes"]))
        source_content.update(copy.deepcopy(
            identity["source_weights_content_sha256"]
        ))
        if contexts[0].minchain:
            shard_cells = identity.get("cb_minchain_cells")
            if not isinstance(shard_cells, Mapping):
                raise ValueError(
                    f"{where}: min-chain shard {index} has no per-cell identity"
                )
            minchain_cells.update(copy.deepcopy(shard_cells))
    merged = build_production_cache_cb_render_identity(
        scope,
        cb_serialization_context=contexts[0],
        col_weights=col_weights,
        render_levers=contract["resolved_levers"],
        render_mechanism_plan=contract["mechanism_plan"],
    )
    merged["source_weights_shapes"] = dict(sorted(source_shapes.items()))
    merged["source_weights_content_sha256"] = dict(
        sorted(source_content.items())
    )
    merged["source_weights_complete"] = True
    merged["source_weights_sha256"] = _combined_source_weights_sha256(
        source_shapes,
        source_content,
    )
    if contexts[0].minchain:
        merged["cb_minchain_cells"] = dict(sorted(minchain_cells.items()))
    validate_cb_render_identity_metadata(
        merged,
        col_weights=col_weights,
        where=f"{where} merged identity",
    )
    return merged


def validate_production_cache_cb_render_identity(
    cache: ProductionWeightCache,
    *,
    expected_context=None,
    expected_qnames: Sequence[str] | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    require_for_formats: Sequence[str] = (),
    require_source_complete: bool = True,
    where: str = "ProductionWeightCache",
):
    """Fail closed on absent, stale, or malformed CB render provenance.

    Consumers rehydrate the stored producer context from this identity.  They
    may compare it with an assignment context, but they never stamp whatever
    CB environment happens to be active when the cache is consumed.
    """
    stored_cb_scope: dict[str, list[str]] = {}
    for qname, fmt in (getattr(cache, "weights", {}) or {}):
        if not _is_cb_format_name(fmt):
            continue
        stored_cb_scope.setdefault(str(qname), []).append(str(fmt))
    required_by_format = any(
        _is_cb_format_name(fmt) for fmt in require_for_formats
    )
    metadata = getattr(cache, "metadata", None)
    identity = (
        metadata.get(CB_RENDER_IDENTITY_METADATA_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    if not stored_cb_scope and not required_by_format and identity is None:
        return None
    if not isinstance(identity, Mapping):
        raise ValueError(
            f"{where}: CB cache is missing versioned "
            f"metadata[{CB_RENDER_IDENTITY_METADATA_KEY!r}]; legacy or "
            "partially resumed CB caches must be rebuilt"
        )
    return validate_cb_render_identity_metadata(
        identity,
        expected_context=expected_context,
        expected_qnames=expected_qnames,
        expected_formats_by_qname=stored_cb_scope,
        col_weights=col_weights,
        require_source_complete=require_source_complete,
        where=where,
    )


def production_cache_cb_render_identity(
    cache: ProductionWeightCache,
    *,
    expected_context=None,
    expected_qnames: Sequence[str] | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    require_for_formats: Sequence[str] = (),
    require_source_complete: bool = True,
    where: str = "ProductionWeightCache",
) -> dict[str, object] | None:
    """Return a validated defensive copy of the cache's stored identity."""
    context = validate_production_cache_cb_render_identity(
        cache,
        expected_context=expected_context,
        expected_qnames=expected_qnames,
        col_weights=col_weights,
        require_for_formats=require_for_formats,
        require_source_complete=require_source_complete,
        where=where,
    )
    if context is None:
        return None
    return copy.deepcopy(
        dict(cache.metadata[CB_RENDER_IDENTITY_METADATA_KEY])
    )


def production_cache_cb_render_provenance(
    cache: ProductionWeightCache,
    *,
    expected_context=None,
    expected_qnames: Sequence[str] | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    require_for_formats: Sequence[str] = (),
    require_source_complete: bool = True,
    where: str = "ProductionWeightCache",
) -> dict[str, object]:
    """Return the exact stored CB provenance fragment for downstream use."""
    identity = production_cache_cb_render_identity(
        cache,
        expected_context=expected_context,
        expected_qnames=expected_qnames,
        col_weights=col_weights,
        require_for_formats=require_for_formats,
        require_source_complete=require_source_complete,
        where=where,
    )
    if identity is None:
        return {}
    return {
        "cb_serialized_payload": copy.deepcopy(
            identity["cb_serialized_payload"]
        ),
        CB_RENDER_IDENTITY_METADATA_KEY: identity,
    }


def validate_cb_render_provenance(
    payload: Mapping[str, object],
    *,
    expected_context=None,
    expected_qnames: Sequence[str] | None = None,
    expected_formats_by_qname: Mapping[
        str, str | Sequence[str]
    ] | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    where: str,
) -> tuple[object, dict[str, object]]:
    """Validate and return a cost/assignment payload's exact CB identity."""
    provenance = payload.get("provenance")
    container = provenance if isinstance(provenance, Mapping) else payload
    identity = container.get(CB_RENDER_IDENTITY_METADATA_KEY)
    context = validate_cb_render_identity_metadata(
        identity if isinstance(identity, Mapping) else None,
        expected_context=expected_context,
        expected_qnames=expected_qnames,
        expected_formats_by_qname=expected_formats_by_qname,
        col_weights=col_weights,
        where=where,
    )
    top_stamp = container.get("cb_serialized_payload")
    from prismaquant.nvfp4_cb_footprint import (
        validate_cb_serialization_context_stamp,
    )

    validate_cb_serialization_context_stamp(
        top_stamp if isinstance(top_stamp, Mapping) else None,
        context,
        where=f"{where} top-level context",
    )
    if top_stamp is None or dict(top_stamp) != dict(
        identity["cb_serialized_payload"]
    ):
        raise ValueError(
            f"{where}: top-level CB context does not exactly match its "
            "value-bearing render identity"
        )
    return context, copy.deepcopy(dict(identity))


def validate_matching_cb_render_identities(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    formats_by_qname: Mapping[str, str | Sequence[str]],
    *,
    where: str,
) -> None:
    """Require equal value-bearing inputs for the requested CB rows."""
    required_scope = _canonical_cb_render_scope(formats_by_qname)
    reference_context = validate_cb_render_identity_metadata(
        reference,
        expected_formats_by_qname=required_scope,
        where=f"{where} reference",
    )
    validate_cb_render_identity_metadata(
        candidate,
        expected_context=reference_context,
        expected_formats_by_qname=required_scope,
        where=f"{where} candidate",
    )
    if reference["render_contract"] != candidate["render_contract"]:
        raise ValueError(f"{where}: resolved CB render contracts differ")
    mismatched: list[str] = []
    mismatched_source: list[str] = []
    for qname in required_scope:
        if (
            list(reference["col_weights_shapes"][qname])
            != list(candidate["col_weights_shapes"][qname])
            or str(reference["col_weights_content_sha256"][qname]).lower()
            != str(candidate["col_weights_content_sha256"][qname]).lower()
        ):
            mismatched.append(qname)
        if (
            list(reference["source_weights_shapes"][qname])
            != list(candidate["source_weights_shapes"][qname])
            or str(reference["source_weights_content_sha256"][qname]).lower()
            != str(candidate["source_weights_content_sha256"][qname]).lower()
        ):
            mismatched_source.append(qname)
    if mismatched:
        raise ValueError(
            f"{where}: CB imatrix values differ for consumed rows; "
            f"sample={mismatched[:8]}"
        )
    if mismatched_source:
        raise ValueError(
            f"{where}: decoded CB source-weight values differ for consumed "
            f"rows; sample={mismatched_source[:8]}"
        )


def _extend_production_cache_cb_render_identity(
    cache: ProductionWeightCache,
    formats_by_qname: Mapping[str, str | Sequence[str]],
    *,
    cb_serialization_context,
    col_weights: Mapping[str, torch.Tensor] | None,
    render_levers: Mapping[str, object] | None = None,
    render_mechanism_plan=None,
) -> None:
    """Validate the existing identity and extend it for a new CB scope."""
    new_scope = _canonical_cb_render_scope(formats_by_qname)
    new_qnames = list(new_scope)
    if not new_scope:
        return
    metadata = dict(getattr(cache, "metadata", {}) or {})
    old_identity = metadata.get(CB_RENDER_IDENTITY_METADATA_KEY)
    old_scope: dict[str, list[str]] = {}
    if old_identity is not None or any(
        _is_cb_format_name(fmt) for _qname, fmt in cache.weights
    ):
        validate_production_cache_cb_render_identity(
            cache,
            expected_context=cb_serialization_context,
            require_for_formats=["NVFP4_CB_K16"],
            require_source_complete=False,
            where="ProductionWeightCache extension",
        )
        old_scope = {
            str(name): [str(fmt) for fmt in formats]
            for name, formats in old_identity["cb_formats_by_qname"].items()
        }
        old_qnames = list(old_scope)
        overlap = sorted(set(old_qnames) & set(new_qnames))
        if overlap:
            _, overlap_shapes, overlap_content = (
                _canonical_cb_col_weights_identity(col_weights, overlap)
            )
            stored_shapes = old_identity["col_weights_shapes"]
            stored_content = old_identity["col_weights_content_sha256"]
            mismatched = [
                name for name in overlap
                if (
                    list(stored_shapes[name]) != overlap_shapes[name]
                    or str(stored_content[name]).lower()
                    != overlap_content[name]
                )
            ]
            if mismatched:
                raise ValueError(
                    "ProductionWeightCache extension: CB col_weights differ "
                    f"for existing render qnames; sample={mismatched[:8]}"
                )
        if all(
            name in old_scope
            and set(formats).issubset(old_scope[name])
            for name, formats in new_scope.items()
        ):
            return
    combined: dict[str, list[str]] = copy.deepcopy(old_scope)
    for name, formats in new_scope.items():
        combined[name] = sorted(set(combined.get(name, ())) | set(formats))
    if old_identity is None and (
        render_levers is None or render_mechanism_plan is None
    ):
        raise ValueError(
            "new ProductionWeightCache CB scope requires explicit resolved "
            "render levers and mechanism plan"
        )
    identity = build_production_cache_cb_render_identity(
        combined,
        cb_serialization_context=cb_serialization_context,
        col_weights=col_weights,
        render_levers=(
            render_levers
            if render_levers is not None
            else old_identity["render_contract"]["resolved_levers"]
        ),
        render_mechanism_plan=(
            render_mechanism_plan
            if render_mechanism_plan is not None
            else old_identity["render_contract"]["mechanism_plan"]
        ),
    )
    if old_identity is not None:
        if identity["render_contract"] != old_identity["render_contract"]:
            raise ValueError(
                "ProductionWeightCache extension: resolved CB render "
                "contract differs from the existing cache"
            )
        identity["source_weights_shapes"] = copy.deepcopy(
            old_identity["source_weights_shapes"]
        )
        identity["source_weights_content_sha256"] = copy.deepcopy(
            old_identity["source_weights_content_sha256"]
        )
        observed = set(identity["source_weights_content_sha256"])
        expected = set(identity["col_weights_qnames"])
        identity["source_weights_complete"] = observed == expected
        identity["source_weights_sha256"] = (
            _combined_source_weights_sha256(
                identity["source_weights_shapes"],
                identity["source_weights_content_sha256"],
            )
            if observed == expected else None
        )
    metadata[CB_RENDER_IDENTITY_METADATA_KEY] = identity
    cache.metadata = metadata


def _render_nvfp4_progressive_candidate(
    *,
    qname: str,
    weight_scaled: torch.Tensor,
    activations_scaled: torch.Tensor | None,
    levers: Mapping[str, object],
    scale_rule: str,
    joint_global_real: torch.Tensor | None,
    act_clip_threshold: float | None,
    act_clip_rescale: str | None,
    fisher_row_weights: torch.Tensor | None,
    include_gptq: bool,
    include_scale_sweep: bool,
) -> torch.Tensor:
    from prismaquant import export_native_compressed as enc

    with _temporary_nvfp4_scale_rule(scale_rule):
        current = enc._rtn_dequant_nvfp4(
            weight_scaled,
            group_size=16,
            global_real_override=joint_global_real,
        )
        if activations_scaled is None or activations_scaled.numel() == 0:
            return current
        if include_gptq:
            if _damp_sweep_enabled():
                current = enc._gptq_obs_rounding_nvfp4_swept(
                    weight_scaled,
                    activations_scaled,
                    group_size=16,
                    global_real_override=joint_global_real,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    static_act_order=bool(
                        levers.get("static_act_order", False)
                    ),
                    joint_scale_opt=bool(
                        levers.get("joint_scale_opt", False)
                    ),
                )
            else:
                current = enc._gptq_obs_rounding_nvfp4(
                    weight_scaled,
                    activations_scaled,
                    group_size=16,
                    damp=enc._resolve_gptq_damp_for_role(qname),
                    global_real_override=joint_global_real,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    static_act_order=bool(
                        levers.get("static_act_order", False)
                    ),
                    joint_scale_opt=bool(
                        levers.get("joint_scale_opt", False)
                    ),
                )
        if include_scale_sweep:
            current = enc._scale_sweep_nvfp4(
                current,
                activations_scaled,
                group_size=16,
                global_real_override=joint_global_real,
                reference_weight=weight_scaled,
                clip_threshold=act_clip_threshold,
                clip_rescale=act_clip_rescale,
                fisher_row_weights=fisher_row_weights,
            )
        return current


def _render_nvfp4_progressively(
    weight: torch.Tensor,
    *,
    qname: str,
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, object],
    joint_global_real: torch.Tensor | None,
    act_clip_threshold: float | None,
    act_clip_rescale: str | None,
    fisher_row_weights: torch.Tensor | None,
    gate_trace: list[dict[str, object]] | None,
) -> torch.Tensor:
    from prismaquant import export_native_compressed as enc

    requested_rule = enc.resolve_nvfp4_scale_rule(
        str(levers.get("nvfp4_scale_rule", "static_6"))
    )
    f6_enabled = requested_rule == enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE
    gptq_enabled = bool(levers.get("gptq", True))
    scale_sweep_enabled = bool(levers.get("scale_sweep", False))
    static_act_order_enabled = bool(
        gptq_enabled and levers.get("static_act_order", False)
    )
    joint_scale_opt_enabled = bool(
        gptq_enabled and levers.get("joint_scale_opt", False)
    )
    gptq_modifiers = tuple(
        name for name, enabled in (
            ("static_act_order", static_act_order_enabled),
            ("joint_scale_opt", joint_scale_opt_enabled),
        )
        if enabled
    )
    gptq_scale_rule = (
        enc.NVFP4_SCALE_RULE_JOINT_MSE
        if joint_scale_opt_enabled
        else None
    )
    min_gain = _env_float(
        "PRISMAQUANT_RENDER_GATE_MIN_GAIN",
        0.0,
        lo=-1.0,
        hi=1.0,
    )

    reference = weight.detach().to(device=weight.device, dtype=torch.float32)
    acts = activations.get(qname)
    acts_for_render = (
        acts.detach().to(device=weight.device, dtype=torch.float32)
        if acts is not None and int(acts.shape[-1]) == int(weight.shape[1])
        else None
    )
    # Score gate candidates on the same clipped/weighted matrix the GPTQ
    # loop optimizes under (audit 2026-07-02 §3.9): the loop passes
    # act_clip_threshold/act_clip_rescale/fisher_row_weights into
    # _activation_matrix_for_gptq, so the gate must too.
    acts_for_gate = _gate_activation_matrix(
        acts_for_render,
        int(weight.shape[1]),
        device=weight.device,
        act_clip_threshold=act_clip_threshold,
        act_clip_rescale=act_clip_rescale,
        fisher_row_weights=fisher_row_weights,
    )
    reference_for_render = reference

    def candidate(
        *,
        label: str,
        scale_rule: str,
        package: tuple[str, ...],
        include_gptq: bool,
        include_scale_sweep: bool,
    ) -> _RenderedCandidate:
        rendered_scaled = _render_nvfp4_progressive_candidate(
            qname=qname,
            weight_scaled=reference_for_render,
            activations_scaled=acts_for_render,
            levers=levers,
            scale_rule=scale_rule,
            joint_global_real=joint_global_real,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=act_clip_rescale,
            fisher_row_weights=fisher_row_weights,
            include_gptq=include_gptq,
            include_scale_sweep=include_scale_sweep,
        )
        rendered = rendered_scaled
        score, metric = _render_score_for_gate(
            reference,
            rendered,
            acts_for_gate,
        )
        return _RenderedCandidate(
            label=label,
            weight=rendered,
            score=float(score),
            metric=metric,
            scale_rule=scale_rule,
            package=package,
            has_gptq=bool(include_gptq),
        )

    static_rule = enc.NVFP4_SCALE_RULE_STATIC_6
    current = candidate(
        label="rtn_static_6",
        scale_rule=static_rule,
        package=(),
        include_gptq=False,
        include_scale_sweep=False,
    )
    if gate_trace is not None:
        gate_trace.append({
            "mechanism": "baseline",
            "selected": current.label,
            "score": float(current.score),
            "metric": current.metric,
            "scale_rule": current.scale_rule,
            "package": list(current.package),
        })

    def apply_gate(
        *,
        mechanism: str,
        candidates: Sequence[_RenderedCandidate],
    ) -> None:
        nonlocal current
        if not candidates:
            return
        best = min(candidates, key=lambda item: item.score)
        decision = gate_render_candidate(
            baseline_score=current.score,
            candidate_score=best.score,
            metric=best.metric,
            min_relative_gain=min_gain,
        )
        accepted = bool(decision.accepted)
        if gate_trace is not None:
            gate_trace.append({
                "mechanism": mechanism,
                "accepted": accepted,
                "selected": best.label if accepted else current.label,
                "candidate": best.label,
                "baseline_score": float(current.score),
                "candidate_score": float(best.score),
                "relative_gain": float(decision.relative_gain),
                "metric": best.metric,
                "reason": str(decision.reason),
                "scale_rule": best.scale_rule,
                "package": list(best.package),
                "candidates": [
                    {
                        "label": cand.label,
                        "score": float(cand.score),
                        "metric": cand.metric,
                        "scale_rule": cand.scale_rule,
                        "package": list(cand.package),
                    }
                    for cand in candidates
                ],
            })
        if accepted:
            old = current.weight
            current = best
            if old is not best.weight:
                del old
        for cand in candidates:
            if cand is not current:
                del cand.weight

    if f6_enabled:
        apply_gate(
            mechanism="four_over_six",
            candidates=[
                candidate(
                    label="four_over_six",
                    scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
                    package=("four_over_six",),
                    include_gptq=False,
                    include_scale_sweep=False,
                )
            ],
        )

    if gptq_enabled and acts_for_render is not None:
        gptq_name = "fisher_gptq" if fisher_row_weights is not None else "gptq"
        primary_scale_rule = gptq_scale_rule or current.scale_rule
        primary_package = (
            (gptq_name, "gptq") if gptq_name != "gptq" else ("gptq",)
        )
        primary_package = tuple(dict.fromkeys((*gptq_modifiers, *primary_package)))
        packages: list[_RenderedCandidate] = [
            candidate(
                label="+".join((primary_scale_rule, *gptq_modifiers, gptq_name)),
                scale_rule=primary_scale_rule,
                package=primary_package,
                include_gptq=True,
                include_scale_sweep=False,
            )
        ]
        if (
            f6_enabled
            and not joint_scale_opt_enabled
            and current.scale_rule != enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE
        ):
            packages.append(candidate(
                label=f"four_over_six+{gptq_name}",
                scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
                package=(
                    (*gptq_modifiers, "four_over_six", gptq_name, "gptq")
                    if gptq_name != "gptq" else
                    (*gptq_modifiers, "four_over_six", "gptq")
                ),
                include_gptq=True,
                include_scale_sweep=False,
            ))
        apply_gate(mechanism=gptq_name, candidates=packages)

    if scale_sweep_enabled and acts_for_render is not None:
        scale_candidates: list[_RenderedCandidate] = [
            candidate(
                label=f"{current.label}+scale_sweep",
                scale_rule=current.scale_rule,
                package=tuple(dict.fromkeys((*current.package, "scale_sweep"))),
                include_gptq=current.has_gptq,
                include_scale_sweep=True,
            )
        ]
        if gptq_enabled and not current.has_gptq:
            gptq_name = "fisher_gptq" if fisher_row_weights is not None else "gptq"
            scale_rule = gptq_scale_rule or current.scale_rule
            pkg = (
                (gptq_name, "gptq", "scale_sweep")
                if gptq_name != "gptq" else
                ("gptq", "scale_sweep")
            )
            pkg = tuple(dict.fromkeys((*gptq_modifiers, *pkg)))
            scale_candidates.append(candidate(
                label="+".join((scale_rule, *gptq_modifiers, gptq_name, "scale_sweep")),
                scale_rule=scale_rule,
                package=pkg,
                include_gptq=True,
                include_scale_sweep=True,
            ))
        if (
            f6_enabled
            and not joint_scale_opt_enabled
            and current.scale_rule != enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE
        ):
            gptq_name = "fisher_gptq" if fisher_row_weights is not None else "gptq"
            include_gptq = bool(gptq_enabled)
            pkg = [*gptq_modifiers, "four_over_six"]
            if include_gptq:
                if gptq_name != "gptq":
                    pkg.extend([gptq_name, "gptq"])
                else:
                    pkg.append("gptq")
            pkg.append("scale_sweep")
            scale_candidates.append(candidate(
                label="+".join(pkg),
                scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
                package=tuple(pkg),
                include_gptq=include_gptq,
                include_scale_sweep=True,
            ))
        apply_gate(mechanism="scale_sweep", candidates=scale_candidates)

    return current.weight.to(device=weight.device, dtype=weight.dtype).contiguous()


def _summarize_render_gate_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "enabled": True,
        "entries": int(len(records)),
        "mechanisms": {},
    }
    mechanisms: dict[str, dict[str, object]] = {}
    for record in records:
        for step in record.get("trace", []):  # type: ignore[union-attr]
            if not isinstance(step, Mapping):
                continue
            mech = str(step.get("mechanism", "unknown"))
            if mech == "baseline":
                continue
            bucket = mechanisms.setdefault(mech, {
                "accepted": 0,
                "rejected": 0,
                "reasons": {},
                "package_accepted": 0,
            })
            accepted = bool(step.get("accepted", False))
            if accepted:
                bucket["accepted"] = int(bucket["accepted"]) + 1
                package = step.get("package")
                if (
                    isinstance(package, Sequence)
                    and not isinstance(package, str)
                    and mech in package
                ):
                    bucket["package_accepted"] = int(bucket["package_accepted"]) + 1
            else:
                bucket["rejected"] = int(bucket["rejected"]) + 1
            reason = str(step.get("reason", "unknown"))
            reasons = bucket["reasons"]
            if isinstance(reasons, dict):
                reasons[reason] = int(reasons.get(reason, 0)) + 1

            package = step.get("package")
            if isinstance(package, Sequence) and not isinstance(package, str):
                for member in package:
                    member_name = str(member)
                    if member_name == mech:
                        continue
                    member_bucket = mechanisms.setdefault(member_name, {
                        "accepted": 0,
                        "rejected": 0,
                        "reasons": {},
                        "package_accepted": 0,
                    })
                    if accepted:
                        member_bucket["package_accepted"] = (
                            int(member_bucket["package_accepted"]) + 1
                        )
    summary["mechanisms"] = mechanisms
    return summary


def render_production_weight(
    weight: torch.Tensor,
    fmt: str,
    *,
    qname: str,
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, object],
    joint_global_real: torch.Tensor | None = None,
    input_global_scale: float | None = None,
    act_clip_threshold: float | None = None,
    act_clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    col_weights: torch.Tensor | None = None,
    cb_serialization_context=None,
    gate_trace: list[dict[str, object]] | None = None,
) -> torch.Tensor:
    """Compute the production-faithful dequantized weight for ``(qname, fmt)``.

    Returns a tensor matching ``weight.shape`` and dtype.  For NVFP4 this
    runs GPTQ + scale_sweep (the activation-aware passes) with the joint
    fused-sibling NVFP4 global if supplied; for BF16 and RTN-only formats
    it falls back to the registry quantize_dequantize because those formats
    don't benefit from activation-aware refinement in the production pipeline.

    ``joint_global_real`` is the max-across-fused-siblings NVFP4 global
    used to keep q/k/v (or gate/up) per-tensor scales unified — same as
    the export's ``_compute_nvfp4_joint_global``.  When ``None`` the
    per-Linear computed value is used (legacy behavior, only correct for
    isolated Linears with no fused siblings).

    ``act_clip_threshold`` is an optional scalar clamp for activation-aware
    render passes. ``fisher_row_weights`` optionally weights local objectives
    by per-token gradient² from h-detail.

    ``col_weights`` is the per-input-column imatrix vector (``(in_features,)``)
    for this ``qname`` — CB **Milestone C** / re-vet **R3**. It is applied for
    the weighted-render families ONLY (``WEIGHTED_RENDER_FAMILIES``: the CB
    codebook families and GGUF), whose exporters ship imatrix-weighted bytes;
    for every other format the argument is inert and the render is
    **bit-identical** whether it is passed or not (pinned by
    ``tests/test_col_weights_render_identity.py``). The vector is supplied by
    the caller rather than derived here from ``activations`` on purpose: the
    exporter's vector is the harvested ``artifacts/cb_col_weights.pkl``, which
    includes the synthesized packed-expert entries
    (``moe_imatrix.synthesize_packed_expert_col_weights``) that no raw
    activation matrix on this path can reproduce. With it, the CB lane's cost,
    KL and shipped bytes can come from ONE render through
    ``ProductionWeightCache``.

    """
    from prismaquant import format_registry as fr

    fmt = fr.canonical_format_name(str(fmt).strip().upper())
    is_cb = fr.get_format(fmt).family in {"nvfp4_cb", "fp8_cb"}
    if is_cb and cb_serialization_context is None:
        raise ValueError(
            f"{qname}={fmt}: production CB render requires an explicit "
            "CBSerializationContext; refusing to use current environment "
            "defaults"
        )
    if is_cb and col_weights is None:
        raise RuntimeError(
            f"{qname}={fmt}: production CB render has no col_weights; "
            "refusing an unweighted cache entry that export cannot reproduce"
        )
    if is_cb and not bool(levers.get("weighted_vq", True)):
        raise RuntimeError(
            f"{qname}={fmt}: weighted_vq cannot be disabled for a production "
            "CB render because export is always imatrix-weighted"
        )
    if col_weights is not None and not (
        _format_supports_render_mechanism(fmt, "weighted_vq")
        and bool(levers.get("weighted_vq", True))
    ):
        # Inert for every non-weighted family: fall through to the exact
        # pre-R3 code path, no branch, no dtype churn.
        col_weights = None
    clip_rescale = "none"
    if str(act_clip_rescale or "none").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
        "none",
    }:
        raise ValueError("activation clip rescaling is not supported")
    progressive_gates = _env_flag("PRISMAQUANT_RENDER_PROGRESSIVE_GATES", True)
    if fmt == "NVFP4" and progressive_gates:
        return _render_nvfp4_progressively(
            weight,
            qname=qname,
            activations=activations,
            levers=levers,
            joint_global_real=joint_global_real,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=clip_rescale,
            fisher_row_weights=fisher_row_weights,
            gate_trace=gate_trace,
        )

    if fmt != "NVFP4":
        spec = fr.get_format(fmt)
        acts = activations.get(qname)
        acts_for_render = (
            acts.detach().to(device=weight.device, dtype=torch.float32)
            if acts is not None and int(acts.shape[-1]) == int(weight.shape[1])
            else None
        )
        # R3 / Milestone C: the weighted families' one render definition,
        # shared with the inline cost render and the emulation path. Calling
        # it for the unweighted case as well is important for CB: it binds the
        # v1/v2 reachable scale set to the same serialization context the
        # cost payload and exporter stamp, while remaining the registry QDQ
        # for every non-CB format.
        if is_cb:
            from prismaquant.nvfp4_cb_footprint import (
                cb_quantize_dequantize_for_context,
            )

            baseline = cb_quantize_dequantize_for_context(
                spec,
                weight.detach(),
                context=cb_serialization_context,
                col_weights=col_weights.reshape(-1).to(weight.device),
                activation_rows=acts_for_render,
            ).to(device=weight.device, dtype=weight.dtype)
        else:
            from prismaquant.emu_forward_kl import weighted_quantize_dequantize

            baseline = weighted_quantize_dequantize(
                spec,
                weight.detach(),
                None if col_weights is None
                else col_weights.reshape(-1).to(weight.device),
            ).to(device=weight.device, dtype=weight.dtype)
        reference = weight.detach().to(torch.float32)
        # Clip-consistent gate matrix — same contract as the NVFP4
        # progressive path (audit 2026-07-02 §3.9): score under the matrix
        # the GPTQ/scale-sweep passes optimize on.
        acts_for_gate = _gate_activation_matrix(
            acts_for_render,
            int(weight.shape[1]),
            device=weight.device,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=clip_rescale,
            fisher_row_weights=fisher_row_weights,
        )
        baseline_score, baseline_metric = _render_score_for_gate(
            reference,
            baseline,
            acts_for_gate,
        )
        current = _RenderedCandidate(
            label=f"{fmt.lower()}+" + (
                "weighted_vq" if col_weights is not None else "rtn"),
            weight=baseline.contiguous(),
            score=float(baseline_score),
            metric=baseline_metric,
            scale_rule="",
            package=(),
            has_gptq=False,
        )
        if gate_trace is not None:
            gate_trace.append({
                "mechanism": "baseline",
                "selected": current.label,
                "score": float(current.score),
                "metric": current.metric,
                "package": [],
            })

        def _apply_non_nv_gate(
            *,
            mechanism: str,
            candidates: Sequence[_RenderedCandidate],
        ) -> None:
            nonlocal current
            if not candidates:
                return
            best = min(candidates, key=lambda item: item.score)
            decision = gate_render_candidate(
                baseline_score=current.score,
                candidate_score=best.score,
                metric=best.metric,
                min_relative_gain=_env_float(
                    "PRISMAQUANT_RENDER_GATE_MIN_GAIN",
                    0.0,
                    lo=-1.0,
                    hi=1.0,
                ),
            )
            if gate_trace is not None:
                gate_trace.append({
                    "mechanism": mechanism,
                    "accepted": bool(decision.accepted),
                    "selected": (
                        best.label if decision.accepted else current.label
                    ),
                    "candidate": best.label,
                    "baseline_score": float(current.score),
                    "candidate_score": float(best.score),
                    "relative_gain": float(decision.relative_gain),
                    "metric": best.metric,
                    "reason": str(decision.reason),
                    "package": list(best.package),
                    "candidates": [
                        {
                            "label": cand.label,
                            "score": float(cand.score),
                            "metric": cand.metric,
                            "package": list(cand.package),
                        }
                        for cand in candidates
                    ],
                })
            if decision.accepted:
                old = current.weight
                current = best
                if old is not best.weight:
                    del old
            for cand in candidates:
                if cand is not current and cand.weight is not current.weight:
                    del cand.weight

        def _non_nv_candidate(
            *,
            label: str,
            weight_dq: torch.Tensor,
            package: tuple[str, ...],
            has_gptq: bool,
        ) -> _RenderedCandidate:
            rendered = weight_dq.to(device=weight.device, dtype=weight.dtype).contiguous()
            score, metric = _render_score_for_gate(
                reference, rendered, acts_for_gate,
            )
            return _RenderedCandidate(
                label=label,
                weight=rendered,
                score=float(score),
                metric=metric,
                scale_rule="",
                package=package,
                has_gptq=bool(has_gptq),
            )

        if (
            _format_supports_render_mechanism(fmt, "gptq")
            and bool(levers.get("gptq", True))
            and acts_for_render is not None
        ):
            from prismaquant import export_native_compressed as enc

            joint_scale_opt = bool(
                levers.get("joint_scale_opt", False)
                and _format_supports_render_mechanism(fmt, "joint_scale_opt")
            )
            static_act_order = bool(
                levers.get("static_act_order", False)
                and _format_supports_render_mechanism(fmt, "static_act_order")
            )
            base_package = (
                ("joint_scale_opt", "gptq")
                if joint_scale_opt else
                ("gptq",)
            )
            use_damp_sweep = (
                _damp_sweep_enabled()
            )

            def _gptq_candidate(use_static_act_order: bool) -> _RenderedCandidate:
                package = tuple(dict.fromkeys((
                    *(
                        ("static_act_order",)
                        if use_static_act_order else
                        ()
                    ),
                    *base_package,
                )))
                if fmt == "MXFP4":
                    if use_damp_sweep:
                        _q, _s, candidate = enc._gptq_obs_rounding_mxfp4_swept(
                            reference,
                            acts_for_render,
                            group_size=32,
                            clip_threshold=act_clip_threshold,
                            clip_rescale=clip_rescale,
                            fisher_row_weights=fisher_row_weights,
                            static_act_order=use_static_act_order,
                        )
                    else:
                        _q, _s, candidate = enc._gptq_obs_rounding_mxfp4(
                            reference,
                            acts_for_render,
                            group_size=32,
                            clip_threshold=act_clip_threshold,
                            clip_rescale=clip_rescale,
                            fisher_row_weights=fisher_row_weights,
                            static_act_order=use_static_act_order,
                        )
                elif use_damp_sweep:
                    _q, _s, candidate = enc._gptq_obs_rounding_fp8_like_swept(
                        reference,
                        acts_for_render,
                        fmt=fmt,
                        group_size=32,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        joint_scale_opt=joint_scale_opt,
                        static_act_order=use_static_act_order,
                    )
                else:
                    _q, _s, candidate = enc._gptq_obs_rounding_fp8_like(
                        reference,
                        acts_for_render,
                        fmt=fmt,
                        group_size=32,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        joint_scale_opt=joint_scale_opt,
                        static_act_order=use_static_act_order,
                    )
                return _non_nv_candidate(
                    label=f"{fmt.lower()}+{'+'.join(package)}",
                    weight_dq=candidate,
                    package=package,
                    has_gptq=True,
                )

            gptq_candidates = [_gptq_candidate(False)]
            if static_act_order:
                gptq_candidates.append(_gptq_candidate(True))
            _apply_non_nv_gate(
                mechanism="gptq",
                candidates=gptq_candidates,
            )

        if (
            _format_supports_render_mechanism(fmt, "scale_sweep")
            and bool(levers.get("scale_sweep", False))
            and acts_for_render is not None
        ):
            if fmt == "MXFP8_E4M3":
                from prismaquant.export_native_compressed import (
                    _mxfp8_scale_sweep_quantize,
                )

                _, _, w_dq = _mxfp8_scale_sweep_quantize(
                    current.weight.detach().to(torch.float32),
                    acts_for_render,
                    group_size=32,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                )
            else:
                from prismaquant.export_native_compressed import (
                    _fp8_dynamic_scale_sweep_quantize,
                )

                _, _, w_dq = _fp8_dynamic_scale_sweep_quantize(
                    current.weight.detach().to(torch.float32),
                    acts_for_render,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                )
            candidate = _non_nv_candidate(
                label=f"{current.label}+scale_sweep",
                weight_dq=w_dq,
                package=tuple(dict.fromkeys((*current.package, "scale_sweep"))),
                has_gptq=current.has_gptq,
            )
            if progressive_gates:
                _apply_non_nv_gate(
                    mechanism="scale_sweep",
                    candidates=[candidate],
                )
                return current.weight.contiguous()
            return candidate.weight.contiguous()
        return current.weight.contiguous()

    from prismaquant.export_native_compressed import _quantize_2d

    with _temporarily_install_act_aware(activations, levers):
        result = _quantize_2d(
            weight.detach().clone(),
            fmt="NVFP4",
            linear_name=qname,
            nvfp4_global_real_override=joint_global_real,
            input_global_scale_override=input_global_scale,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=clip_rescale,
            fisher_row_weights=fisher_row_weights,
            compute_only=True,
        )
    w_dq = result["_w_dq"]
    return w_dq.to(device=weight.device, dtype=weight.dtype).contiguous()


def _resolve_production_render_levers(
    levers: Mapping[str, object] | None,
) -> dict:
    """Normalize a caller lever dict to the production render contract.

    Shared by ``fill_production_weight_cache`` (resident) and the streaming
    driver so both apply IDENTICAL damp/JSO/act-order/scale-rule defaulting —
    the rendered weights must be byte-identical between the two paths.
    """
    levers = dict(levers) if levers is not None else {}
    default_optional_levers = not bool(levers.pop("none", False))
    if not default_optional_levers:
        for name in (
            "gptq",
            "scale_sweep",
            "fisher_gptq",
            "static_act_order",
            "joint_scale_opt",
        ):
            levers.setdefault(name, False)
    levers.setdefault("gptq", True)
    levers.setdefault(
        "gptq_damp_sweep",
        bool(levers.get("gptq", True))
        and _damp_sweep_enabled(),
    )
    if not levers["gptq_damp_sweep"]:
        # Provenance: which fixed damp the no-sweep renders used
        # (PRISMAQUANT_GPTQ_DAMP, default 0.01) — without this, two
        # fixed-damp caches are metadata-indistinguishable.
        from prismaquant.export_native_compressed import (
            _resolve_gptq_fixed_damp,
        )
        levers.setdefault("gptq_fixed_damp", _resolve_gptq_fixed_damp())
    levers.setdefault("scale_sweep", False)
    levers.setdefault(
        "static_act_order",
        _env_flag("PRISMAQUANT_GPTQ_STATIC_ACT_ORDER", False),
    )
    levers.setdefault(
        "joint_scale_opt",
        _env_flag("PRISMAQUANT_NVFP4_JOINT_SCALE_OPT", False),
    )
    if not bool(levers.get("gptq", True)):
        levers["static_act_order"] = False
        levers["joint_scale_opt"] = False
    levers.setdefault(
        "fisher_gptq",
        _env_flag("PRISMAQUANT_FISHER_WEIGHTED_GPTQ", False),
    )
    from prismaquant.export_native_compressed import (
        NVFP4_SCALE_RULE_ENV,
        NVFP4_SCALE_RULE_JOINT_MSE,
        resolve_nvfp4_scale_rule,
    )
    if (
        bool(levers.get("joint_scale_opt", False))
        and "nvfp4_scale_rule" not in levers
        and NVFP4_SCALE_RULE_ENV not in os.environ
    ):
        levers["nvfp4_scale_rule"] = NVFP4_SCALE_RULE_JOINT_MSE
    levers.setdefault("nvfp4_scale_rule", resolve_nvfp4_scale_rule())
    return levers


def _resolve_render_mechanism_plan(levers: Mapping[str, object]):
    """Build the render-mechanism ordering plan from normalized levers."""
    enabled_mechanisms: list[str] = []
    if str(levers.get("nvfp4_scale_rule", "")).strip() == "four_over_six_mse":
        enabled_mechanisms.append("four_over_six")
    if bool(levers.get("gptq", True)):
        enabled_mechanisms.append("gptq")
    if bool(levers.get("static_act_order", False)):
        enabled_mechanisms.append("static_act_order")
    if bool(levers.get("joint_scale_opt", False)):
        enabled_mechanisms.append("joint_scale_opt")
    if bool(levers.get("fisher_gptq", False)):
        enabled_mechanisms.append("fisher_gptq")
    if bool(levers.get("scale_sweep", False)):
        enabled_mechanisms.append("scale_sweep")
    mechanism_plan = resolve_render_mechanism_order(enabled_mechanisms)
    if mechanism_plan.errors:
        raise ValueError(
            "invalid render mechanism plan: " + "; ".join(mechanism_plan.errors)
        )
    return mechanism_plan


def fill_production_weight_cache(
    model: nn.Module,
    calib_ids: torch.Tensor,
    qnames: Sequence[str],
    *,
    formats: Sequence[str] = ("NVFP4",),
    render_assignment: Mapping[str, str] | None = None,
    levers: Mapping[str, bool] | None = None,
    max_act_rows: int = 256,
    progress: bool = True,
    cache_dir: str | Path | None = None,
    recache_pass: bool = False,
    recache_assignment: Mapping[str, str] | None = None,
    recache_profile=None,
    recache_include_activation_quant: bool = True,
    recache_microbatch_size: int = 1,
    h_detail_dir: str | Path | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    cb_serialization_context=None,
) -> ProductionWeightCache:
    """End-to-end fill: collect activations, render production δw per
    (qname, fmt), return a `ProductionWeightCache`.

    Args:
      model: live HF model on the export device.
      calib_ids: ``[N, T]`` token id tensor for activation collection.
      qnames: which Linears are eligible to render (skips MoE packed
        experts; handle those separately via `_quantize_3d_packed`
        extensions).
      formats: which formats to pre-render when `render_assignment` is not
        supplied.
      render_assignment: optional concrete export assignment. When supplied,
        render exactly the non-BF16 `(qname, fmt)` entries used by that
        assignment instead of the full `qnames x formats` menu.
      levers: which production levers to enable (default: GPTQ with optional
        joint NVFP4 scale optimization when requested by the caller).
      recache_pass: when True, run a second calibration forward with the
        concrete production assignment installed from this cache and refit
        ``activation_max_abs`` under quantized upstream weights.
      recache_assignment: required when ``recache_pass`` is True.  Candidate
        caches with multiple possible formats per Linear are ambiguous; recache
        needs the actual export assignment.
      h_detail_dir: optional probe h-detail directory retained for archived
        Fisher ablations. V1 production defaults do not require it.
      col_weights: optional ``{qname: (in_features,)}`` imatrix map (re-vet R3
        / CB Milestone C). Applied only to the weighted-render families
        (``WEIGHTED_RENDER_FAMILIES``); every other format's rendered bytes are
        bit-identical whether it is supplied or not. This is what lets the CB
        lane's allocator cost, frontier KL and shipped bytes all come from ONE
        ``ProductionWeightCache`` render instead of a separate
        skeleton-requantize path.
      cb_serialization_context: explicit artifact serialization identity for
        every CB render. Required when the scope contains a CB format; never
        inferred from the current environment.
    """
    if recache_pass and not recache_assignment:
        raise ValueError(
            "recache_pass=True requires recache_assignment with the concrete "
            "production assignment"
        )
    levers = _resolve_production_render_levers(levers)
    mechanism_plan = _resolve_render_mechanism_plan(levers)
    if progress and mechanism_plan.ordered:
        print(
            "[prod-cache] render mechanism order: "
            + " -> ".join(spec.name for spec in mechanism_plan.ordered),
            flush=True,
        )

    from prismaquant import format_registry as fr

    def _canon(fmt: str) -> str:
        return fr.canonical_format_name(str(fmt).strip().upper())

    requested_formats = tuple(
        dict.fromkeys(_canon(f) for f in formats if str(f).strip())
    )
    eligible_qnames = set(qnames)
    if render_assignment is not None:
        render_formats_by_qname: dict[str, tuple[str, ...]] = {}
        for qname, fmt in render_assignment.items():
            q = str(qname)
            if q not in eligible_qnames:
                continue
            fmt_canon = _canon(fmt)
            if fmt_canon == "BF16":
                continue
            render_formats_by_qname[q] = (fmt_canon,)
        qname_set = set(render_formats_by_qname)
        render_scope = "assignment"
    else:
        non_bf16_formats = tuple(
            f for f in requested_formats if f != "BF16"
        )
        render_formats_by_qname = {
            q: non_bf16_formats for q in eligible_qnames
        }
        qname_set = {
            q for q, fmts in render_formats_by_qname.items() if fmts
        }
        render_scope = "format-menu"

    cb_render_identity = build_production_cache_cb_render_identity(
        render_formats_by_qname,
        cb_serialization_context=cb_serialization_context,
        col_weights=col_weights,
        render_levers=levers,
        render_mechanism_plan=mechanism_plan,
    )
    if cb_render_identity is not None:
        named_modules = dict(model.named_modules())
        cb_source_weights = {
            qname: named_modules[qname].weight.detach()
            for qname in cb_render_identity["col_weights_qnames"]
            if qname in named_modules
            and hasattr(named_modules[qname], "weight")
        }
        cb_render_identity = bind_cb_render_identity_source_weights(
            cb_render_identity,
            cb_source_weights,
            require_complete=True,
            where="ProductionWeightCache dense source binding",
        )

    if not qname_set:
        return ProductionWeightCache(
            weights={},
            levers=dict(levers),
            metadata={
                "render_scope": render_scope,
                "requested_formats": list(requested_formats),
                "requested_entries": 0,
                **({
                    CB_RENDER_IDENTITY_METADATA_KEY: cb_render_identity,
                } if cb_render_identity is not None else {}),
            },
        )
    model_profile = recache_profile
    if model_profile is None:
        try:
            from .model_profiles import profile_from_model
            model_profile = profile_from_model(model)
        except Exception:
            model_profile = None

    if progress:
        requested_entries = sum(
            len(fmts) for fmts in render_formats_by_qname.values()
        )
        print(f"[prod-cache] levers={dict(sorted(levers.items()))}", flush=True)
        print(
            f"[prod-cache] render_scope={render_scope} "
            f"qnames={len(qname_set)} entries={requested_entries}",
            flush=True,
        )

    # RESUME: when disk-streaming is on and prior shards exist, only
    # collect activations for Linears whose shards we still need to
    # render.  On a job that's 99%+ complete this drops activation
    # collection memory + compute by 99% — and lets a borderline-OOM
    # job finish on the same hardware.
    cache_dir_path: Path | None = None
    if cache_dir is not None:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        stale_cb_shards = [
            cache_dir_path / _cache_weight_filename(qname, fmt)
            for qname, fmts in render_formats_by_qname.items()
            for fmt in fmts
            if _is_cb_format_name(fmt)
            and (cache_dir_path / _cache_weight_filename(qname, fmt)).is_file()
        ]
        if stale_cb_shards:
            raise RuntimeError(
                "production CB cache resume is disabled because existing "
                "weight shards are not independently bound to the fresh "
                "serialization/imatrix identity; move or rebuild the cache "
                f"directory first. sample={stale_cb_shards[:8]}"
            )
    render_score_sidecar_path: Path | None = (
        cache_dir_path / "render_scores.json"
        if cache_dir_path is not None else None
    )
    render_score_records: dict[str, dict[str, object]] = (
        _load_render_score_sidecar(render_score_sidecar_path)
    )
    # CB weight shards are deliberately never resumed: their rendered values
    # are only trustworthy under the fresh source/imatrix/context identity
    # constructed above.  The score sidecar is another derived artifact of
    # that same render.  Drop any prior score for the fresh CB scope as well,
    # otherwise a failed re-render could leave an old score looking current
    # even though no matching CB shard was admitted.
    fresh_cb_score_keys = {
        _render_score_record_key(qname, fmt)
        for qname, fmts in render_formats_by_qname.items()
        for fmt in fmts
        if _is_cb_format_name(fmt)
    }
    for score_key in fresh_cb_score_keys:
        render_score_records.pop(score_key, None)
    if progress and render_score_records:
        print(
            f"[prod-cache] resume: loaded {len(render_score_records)} "
            "render-score entries from sidecar",
            flush=True,
        )

    fmt_set = {
        fmt
        for fmts in render_formats_by_qname.values()
        for fmt in fmts
    }
    render_base_fmt_set = {_render_base_format(fmt) for fmt in fmt_set}
    # Store activations for every missing rendered format.  NVFP4 needs them
    # for GPTQ/JSO; FP8_DYNAMIC/FP8_E4M3 and explicit MX formats need them
    # for their activation-aware renders, and the production-render allocator
    # cost always needs them to score the final local forward error after the
    # format's activation quantizer.
    activation_aware_formats = set(fmt_set)
    qnames_to_render: set[str] = set(qname_set)
    missing_formats_by_qname: dict[str, set[str]] = {
        q: set(render_formats_by_qname.get(q, ())) for q in qname_set
    }
    if cache_dir_path is not None:
        # A qname is FULLY done if every requested format has a shard.
        prerendered = 0
        for q in list(qname_set):
            missing = {
                f for f in render_formats_by_qname.get(q, ())
                if not (cache_dir_path / _cache_weight_filename(q, f)).is_file()
            }
            missing_formats_by_qname[q] = missing
            if not missing:
                qnames_to_render.discard(q)
                prerendered += 1
        if progress and prerendered:
            print(
                f"[prod-cache] resume: {prerendered} qnames already on disk "
                f"({len(qnames_to_render)} still need rendering)",
                flush=True,
            )
    qnames_needing_activation = set()
    for q, missing in missing_formats_by_qname.items():
        requested = tuple(render_formats_by_qname.get(q, ()))
        missing_activation_render = any(
            f in activation_aware_formats for f in missing
        )
        missing_activation_score = any(
            f in activation_aware_formats
            and _render_score_record_key(q, f) not in render_score_records
            for f in requested
        )
        if missing_activation_render or missing_activation_score:
            qnames_needing_activation.add(q)
    device = next(model.parameters()).device
    activation_store_device = (
        device if device.type == "cuda" else torch.device("cpu")
    )
    activation_store_dtype = torch.float32
    if progress and qnames_needing_activation:
        print(
            f"[prod-cache] activation_capture "
            f"store_device={activation_store_device} "
            f"store_dtype={activation_store_dtype} "
            f"qnames={len(qnames_needing_activation)}",
            flush=True,
        )
    # RESUME: if all qnames are already rendered AND we have either a
    # sidecar OR no need for max_abs (no NVFP4 in formats), skip the
    # forward pass entirely.  Avoids OOM from the model's forward pass
    # itself on big models (e.g. linear-attention torch fallback can
    # spike memory mid-pass on Qwen3.5/3.6 27B+).
    sidecar_path: Path | None = (
        cache_dir_path / "activation_max_abs.json"
        if cache_dir_path is not None else None
    )
    skip_forward = (
        cache_dir_path is not None
        and not qnames_needing_activation
        and (
            (sidecar_path is not None and sidecar_path.is_file())
            or "NVFP4" not in render_base_fmt_set
        )
    )
    collector = None  # may stay None on the skip_forward path
    if skip_forward:
        if progress:
            print(
                "[prod-cache] resume: all qnames pre-rendered + max_abs "
                "available, skipping activation forward pass",
                flush=True,
            )
        activations: dict[str, torch.Tensor] = {}
    else:
        # Hook every relevant Linear so we always get max_abs (cheap), but
        # only STORE full activations for Linears we still need to render.
        collector = _LinearActivationCollector(
            model,
            qnames=qname_set,
            max_rows=max_act_rows,
            store_qnames=qnames_needing_activation,
            store_device=activation_store_device,
            store_dtype=activation_store_dtype,
            profile=model_profile,
        )
        collector.install()
        try:
            with torch.no_grad():
                for i in range(calib_ids.size(0)):
                    batch = calib_ids[i:i + 1].to(device)
                    try:
                        model(batch, use_cache=False)
                    except TypeError:
                        # Some non-HF or older model wrappers do not expose
                        # use_cache. The cache is only an inference speed
                        # feature; activation collection is still correct
                        # without the explicit flag on those models.
                        model(batch)
        finally:
            collector.remove()
        activations = collector.collected()

    if device.type == "cuda" and qnames_needing_activation:
        cpu_activations = [
            name for name, acts in activations.items()
            if acts.device.type != "cuda"
        ]
        if cpu_activations:
            raise RuntimeError(
                "production cache captured non-CUDA activations for "
                f"{len(cpu_activations)} Linears; sample={cpu_activations[:3]}"
            )

    if progress:
        activation_bytes = sum(
            int(t.numel()) * int(t.element_size())
            for t in activations.values()
        )
        activation_devices = sorted({str(t.device) for t in activations.values()})
        print(
            f"[prod-cache] collected activations for "
            f"{len(activations)}/{len(qname_set)} Linears "
            f"resident_bytes={activation_bytes:,} "
            f"devices={activation_devices}",
            flush=True,
        )

    weights: dict[tuple[str, str], object] = {}
    failed: dict[tuple[str, str], str] = {}
    qname_to_module: dict[str, nn.Module] = {}

    if cache_dir_path is not None and progress:
        print(f"[prod-cache] streaming cache to {cache_dir_path}/", flush=True)

    for full_name, mod, attr in iter_quantizable_tensors(model, model_profile):
        if attr != "weight" or not isinstance(mod, nn.Linear):
            continue
        qname = full_name[:-7] if full_name.endswith(".weight") else full_name
        if qname in qname_set:
            qname_to_module[qname] = mod

    fused_sibling_mapping = (
        _fused_sibling_leaf_mapping_from_profile(model_profile)
        if model_profile is not None
        else {}
    )
    fisher_rows = (
        _FisherRowWeightCache(
            h_detail_dir,
            fused_sibling_mapping or None,
        )
        if (bool(levers.get("fisher_gptq", False)) and h_detail_dir)
        else None
    )
    if progress and bool(levers.get("fisher_gptq", False)):
        if fisher_rows is None:
            print(
                "[prod-cache] Fisher weighting requested but no h_detail_dir "
                "was provided; falling back to unweighted objectives",
                flush=True,
            )
        else:
            print(
                f"[prod-cache] Fisher weighting using h-detail dir "
                f"{fisher_rows.detail_dir}",
                flush=True,
            )

    # HIGH-1: compute joint NVFP4 fused-sibling globals so q/k/v share a
    # per-tensor scale (and gate/up likewise), matching the export's
    # `_compute_nvfp4_joint_global` behavior.  Without this each sibling
    # gets its own scale and vLLM's loader either rejects the artifact or
    # silently runs with degraded accuracy.
    joint_globals: dict[str, torch.Tensor] = {}
    needs_nvfp4_render = any(
        any(_render_base_format(fmt) == "NVFP4" for fmt in missing)
        for missing in missing_formats_by_qname.values()
    )
    if needs_nvfp4_render:
        from prismaquant.export_native_compressed import (
            _compute_nvfp4_joint_global,
        )
        synthetic_assignment = {q: "NVFP4" for q in qname_to_module}
        joint_globals = _compute_nvfp4_joint_global(
            model,
            synthetic_assignment,
            profile=model_profile,
        )
        if progress:
            print(
                f"[prod-cache] computed joint NVFP4 globals for "
                f"{len(joint_globals)} fused-sibling members",
                flush=True,
            )

    # MED-3: per-Linear calibrated max_abs used by the export's act-clip
    # step.  For fused-sibling groups the value is unified (max across
    # siblings), matching the export's joint input_global_scale derivation.
    # We store max_abs directly (not 6/max_abs) — see ProductionWeightCache
    # docstring on the convention difference.
    activation_max_abs: dict[str, float] = {}

    # RESUME: load previously-computed max_abs values from the sidecar
    # JSON if disk-streaming + sidecar exists.  Lets a resume run skip
    # both activation collection and max_abs recomputation for already-
    # rendered qnames.  ``sidecar_path`` was defined earlier (before the
    # forward-skip decision); re-using it here.
    if sidecar_path is not None and sidecar_path.is_file():
        import json as _json
        try:
            activation_max_abs.update(_json.loads(sidecar_path.read_text()))
            if progress:
                print(
                    f"[prod-cache] resume: loaded {len(activation_max_abs)} "
                    f"max_abs entries from sidecar",
                    flush=True,
                )
        except Exception as e:
            if progress:
                print(
                    f"[prod-cache] sidecar load failed ({e}); recomputing",
                    flush=True,
                )

    if "NVFP4" in render_base_fmt_set:
        # Group by fused sibling key for max-across-siblings unification.
        from prismaquant.decision_units import fused_group_key

        per_qname_max_abs: dict[str, float] = {}
        for qname, _ in qname_to_module.items():
            # 1. Sidecar (resume) wins — these are the pre-computed values
            #    from a prior run.
            if qname in activation_max_abs:
                per_qname_max_abs[qname] = activation_max_abs[qname]
                continue
            # 2. Collector's per-Linear scalar (always populated for
            #    Linears that were hooked, even if no full activation
            #    tensor was stored).  ``collector`` is None on the
            #    skip_forward path, in which case we can only fall
            #    through to the activations-tensor path (which is
            #    empty on skip_forward, so we just continue).
            mx = (
                collector.max_abs.get(qname, 0.0)
                if collector is not None else 0.0
            )
            if mx <= 0:
                a = activations.get(qname)
                if a is None:
                    continue
                mx = float(a.abs().max().item())
            if mx <= 0:
                continue
            per_qname_max_abs[qname] = mx

        # Unify across fused sibling groups by taking the max.
        groups: dict[str, list[str]] = {}
        for qname in per_qname_max_abs:
            gk = (
                fused_group_key(model_profile, qname)
                if model_profile is not None else qname
            )
            groups.setdefault(gk, []).append(qname)
        for gk, members in groups.items():
            shared = max(per_qname_max_abs[m] for m in members)
            for m in members:
                activation_max_abs[m] = shared
        if progress and activation_max_abs:
            print(
                f"[prod-cache] computed activation max_abs for "
                f"{len(activation_max_abs)} Linears "
                f"({len(groups)} fused groups)",
                flush=True,
            )
        # Persist max_abs to sidecar so future resume runs can skip
        # activation collection entirely for completed qnames.
        if sidecar_path is not None and activation_max_abs:
            import json as _json
            sidecar_path.write_text(_json.dumps(activation_max_abs, indent=2))

    # MEM: free per-Linear activation tensors after each render so peak
    # memory stays bounded.  On 27B, 497 Linears × ~10K in_features × 512
    # rows × fp32 = ~10 GB just for activations.  Freeing in-loop drops
    # this to ~20 MB resident.
    import gc as _gc
    activations_local = dict(activations)  # shallow copy; we'll pop entries
    n = sum(len(render_formats_by_qname.get(q, ())) for q in qname_to_module)
    done = 0
    skipped_resumed = 0
    skipped_prewritten = 0
    render_gate_records: list[dict[str, object]] = []
    for qname, mod in qname_to_module.items():
        weight = mod.weight.data
        joint = joint_globals.get(qname)
        max_abs = activation_max_abs.get(qname)
        row_weights = (
            fisher_rows.get(qname)
            if bool(levers.get("fisher_gptq", False))
            and fisher_rows is not None
            else None
        )
        # _quantize_2d's input_global_scale_override expects the export
        # convention.  It only affects emitted metadata in compute_only
        # mode (not the dequantized weight values), but we pass the
        # correct convention so the metadata is honest in case future
        # code consumes it.  Routed through the export helper so it
        # tracks PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE (audit C1).
        from prismaquant.export_native_compressed import (
            _nvfp4_input_global_scale_from_max_abs,
        )
        export_scale = (
            _nvfp4_input_global_scale_from_max_abs(float(max_abs))
            if (max_abs is not None and max_abs > 0)
            else None
        )
        for fmt in render_formats_by_qname.get(qname, ()):
            fmt_key = str(fmt).upper()
            render_fmt = _render_base_format(fmt_key)
            key = (qname, fmt_key)
            if key in weights:
                skipped_prewritten += 1
                done += 1
                if progress and done % 25 == 0:
                    print(f"[prod-cache] {done}/{n}", flush=True)
                continue
            # RESUME: in disk-streaming mode, if a shard already exists
            # for (qname, fmt) on disk, treat it as previously rendered
            # and skip re-rendering.  This lets a job that OOM'd at 95%
            # resume without re-doing the work — just rebuild the manifest
            # from the surviving .pt files.
            if cache_dir_path is not None:
                fname = _cache_weight_filename(qname, fmt_key)
                disk_path = cache_dir_path / fname
                if disk_path.is_file():
                    if _is_cb_format_name(fmt_key):
                        raise RuntimeError(
                            "production CB cache resume is disabled for "
                            f"{qname}@{fmt_key}: pre-existing shard "
                            f"{disk_path} has no independently verifiable "
                            "render identity"
                        )
                    weights[(qname, fmt_key)] = fname
                    skipped_resumed += 1
                    score_key = _render_score_record_key(qname, fmt_key)
                    if score_key not in render_score_records:
                        try:
                            cached = torch.load(
                                disk_path,
                                map_location=weight.device,
                                weights_only=True,
                            ).to(device=weight.device, dtype=weight.dtype)
                            render_score_records[score_key] = _render_score_record(
                                qname=qname,
                                fmt=fmt_key,
                                render_format=render_fmt,
                                reference_weight=weight,
                                rendered_weight=cached,
                                activations=activations_local.get(qname),
                                activation_max_abs=max_abs,
                            )
                            del cached
                        except Exception:
                            pass
                    # Do NOT pop activations_local[qname] here: this
                    # loop iterates through every format for this
                    # Linear, and a later format in the same outer
                    # iteration may still need the activation tensor
                    # to render.  The outer pop after the format loop
                    # drops it once all formats are done.
                    continue
            try:
                gate_trace: list[dict[str, object]] = []
                w_dq = render_production_weight(
                    weight, render_fmt,
                    qname=qname,
                    activations=activations_local,
                    levers=levers,
                    joint_global_real=joint,
                    input_global_scale=export_scale,
                    fisher_row_weights=row_weights,
                    col_weights=(
                        None if col_weights is None
                        else col_weights.get(qname)
                    ),
                    cb_serialization_context=cb_serialization_context,
                    gate_trace=gate_trace,
                )
                render_score_records[_render_score_record_key(qname, fmt_key)] = (
                    _render_score_record(
                        qname=qname,
                        fmt=fmt_key,
                        render_format=render_fmt,
                        reference_weight=weight,
                        rendered_weight=w_dq,
                        activations=activations_local.get(qname),
                        activation_max_abs=max_abs,
                    )
                )
                if gate_trace:
                    render_gate_records.append({
                        "qname": qname,
                        "format": fmt_key,
                        "render_format": render_fmt,
                        "trace": gate_trace,
                    })
            except Exception as e:
                failed[(qname, fmt_key)] = str(e)
                if progress:
                    print(
                        f"[prod-cache] FAILED {qname} @ {fmt}: {e}",
                        flush=True,
                    )
                continue
            # MEM: store as the model's native dtype (bf16 by default)
            # rather than fp32 — _quantize_2d's compute_only path returns
            # fp32 but we always re-cast at install time, so storing fp32
            # is wasteful (2× memory).  On 27B this drops the cache from
            # ~25 GB to ~12 GB.
            _store_rendered_weight_entry(
                weights=weights,
                cache_dir_path=cache_dir_path,
                qname=qname,
                fmt=fmt_key,
                tensor=w_dq,
                weight_dtype=weight.dtype,
            )
            done += 1
            del w_dq
            if progress and done % 25 == 0:
                print(f"[prod-cache] {done}/{n}", flush=True)
                _write_render_score_sidecar(
                    render_score_sidecar_path,
                    render_score_records,
                )
        # Free this Linear's activation tensor — won't render this qname
        # again, and the activation can be tens of MB on big models.
        activations_local.pop(qname, None)
        if done % 50 == 0:
            _gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
    if progress:
        print(
            f"[prod-cache] rendered {len(weights)} (qname, fmt) entries "
            f"({skipped_resumed} resumed from disk, "
            f"{skipped_prewritten} pre-existing entries); "
            f"{len(failed)} failures",
            flush=True,
        )
    _write_render_score_sidecar(render_score_sidecar_path, render_score_records)
    render_gate_summary = _summarize_render_gate_records(render_gate_records)
    cache = ProductionWeightCache(
        weights=weights,
        levers=dict(levers),
        activation_max_abs=activation_max_abs or None,
        failed=failed,
        cache_dir=str(cache_dir_path) if cache_dir_path is not None else None,
        metadata={
            "render_mechanism_order": [
                {
                    "name": spec.name,
                    "operation": spec.operation,
                    "scope": spec.scope,
                    "gate_metric": spec.gate_metric,
                }
                for spec in mechanism_plan.ordered
            ],
            "render_failures": {
                f"{qname}|{fmt}": str(error)
                for (qname, fmt), error in sorted(failed.items())
            },
            "render_gates": {
                **render_gate_summary,
                "records": render_gate_records,
            },
            "render_scores": {
                "schema": "prismaquant.production_render_scores.v1",
                "entries": int(len(render_score_records)),
                "records": dict(sorted(render_score_records.items())),
                "cost_semantics": (
                    "score is the rendered candidate's local forward-error "
                    "mean after the format activation quantizer; score_sum "
                    "is score multiplied by activation_rows * out_features "
                    "for output metrics, or by parameter count for weight_mse "
                    "fallback. raw_render_score records the post-render "
                    "weight-only reconstruction objective used by local "
                    "render gates."
                ),
            },
            "four_over_six": (
                render_gate_summary.get("mechanisms", {}).get("four_over_six", {
                    "accepted": 0,
                    "rejected": 0,
                    "package_accepted": 0,
                    "reasons": {},
                })
                if isinstance(render_gate_summary.get("mechanisms"), dict)
                else {
                    "accepted": 0,
                    "rejected": 0,
                    "package_accepted": 0,
                    "reasons": {},
                }
            ),
            "fisher_weighted_gptq": {
                "enabled": bool(levers.get("fisher_gptq", False)),
                "h_detail_dir": str(h_detail_dir) if h_detail_dir else None,
                "loaded": (
                    int(fisher_rows.loads)
                    if fisher_rows is not None
                    and bool(levers.get("fisher_gptq", False))
                    else 0
                ),
                "misses": (
                    int(fisher_rows.misses)
                    if fisher_rows is not None
                    and bool(levers.get("fisher_gptq", False))
                    else 0
                ),
            },
            "render_scope": render_scope,
            "requested_formats": list(requested_formats),
            "requested_entries": int(n),
            **({
                CB_RENDER_IDENTITY_METADATA_KEY: cb_render_identity,
            } if cb_render_identity is not None else {}),
        },
    )
    if recache_pass:
        from prismaquant.production_recache import recache_production_weight_cache

        if progress:
            print("[prod-cache] running production activation re-cache", flush=True)
        cache.prefetch_assignment(
            recache_assignment or {},
            max_resident_bytes=(
                cache._lru_max_bytes if cache._lru_max_bytes > 0 else None
            ),
            max_workers=4,
            require=False,
            progress=progress,
        )
        recache_production_weight_cache(
            model,
            calib_ids,
            recache_assignment or {},
            cache,
            profile=model_profile,
            include_activation_quant=recache_include_activation_quant,
            microbatch_size=recache_microbatch_size,
            progress=progress,
        )
        compacted = cache.compact_for_pickle()
        if progress and compacted:
            print(
                f"[prod-cache] compacted {compacted} resident cache tensors "
                "back to path references after re-cache",
                flush=True,
            )
    return cache


class _PackedExpertActivationCollector:
    """Capture module-level input ``X`` for each packed-experts module.

    Packed 3-D MoE experts are not ``nn.Linear`` and so are invisible to
    ``_LinearActivationCollector``.  This collector hooks the experts module
    itself and reservoir-samples its module-level input (the pre-routing token
    hidden states ``[*, hidden]``).  At render time the captured ``X`` is split
    into per-expert routed rows by ``derive_per_expert_activations`` — the same
    routing the live forward used — so each expert's GPTQ Hessian sees exactly
    the rows that route to it.

    The per-module token budget (``module_token_budget``) is intentionally much
    larger than the per-Linear ``max_rows`` budget: with top-k routing over E
    experts each expert only sees ~``top_k/E`` of the tokens, so a stable
    per-expert Hessian needs ``~max_rows_per_expert * E / top_k`` module tokens.
    """

    def __init__(
        self,
        model: nn.Module,
        experts_qnames: set[str],
        *,
        module_token_budget: int,
        store_device: torch.device | str,
        store_dtype: torch.dtype = torch.float32,
        profile=None,
    ):
        self.model = model
        self.profile = profile
        self.experts_qnames = set(experts_qnames)
        self.module_token_budget = int(module_token_budget)
        self.store_device = torch.device(store_device)
        self.store_dtype = store_dtype
        self.activations: dict[str, list[torch.Tensor]] = {}
        self._priorities: dict[str, torch.Tensor] = {}
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(1234)
        self._handles: list = []
        self._modules_by_qname: dict[str, nn.Module] = {}
        from prismaquant.sensitivity_probe import _is_packed_experts_module
        for qname, mod in model.named_modules():
            if not _is_packed_experts_module(mod, self.profile):
                continue
            if qname not in self.experts_qnames:
                continue
            self._modules_by_qname[qname] = mod
            self.activations[qname] = []

    def install(self) -> None:
        for qname, mod in self._modules_by_qname.items():
            self._handles.append(
                mod.register_forward_pre_hook(self._make_hook(qname))
            )

    def _make_hook(self, qname: str):
        def hook(module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return
            x = args[0]
            # NOT non_blocking: async D2H into pageable memory can read
            # the tensor before the producing kernel finishes under GPU
            # contention, deterministically corrupting the snapshot to NaN
            # (2026-07-11, inline-export agent repro).
            flat = x.detach().reshape(-1, x.shape[-1]).to(
                device=self.store_device,
                dtype=self.store_dtype,
            )
            current = (
                torch.cat(self.activations[qname], dim=0)
                if self.activations[qname]
                else None
            )
            sampled, priorities = update_priority_reservoir(
                current,
                self._priorities.get(qname),
                flat,
                max_rows=self.module_token_budget,
                generator=self._gen,
            )
            self.activations[qname] = [] if sampled is None else [sampled]
            if priorities is None:
                self._priorities.pop(qname, None)
            else:
                self._priorities[qname] = priorities
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def collected(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for qname, parts in self.activations.items():
            if parts:
                out[qname] = torch.cat(parts, dim=0)
        return out


def fill_packed_expert_cache_entries(
    cache: ProductionWeightCache,
    model: nn.Module,
    calib_ids: torch.Tensor | None,
    *,
    render_assignment: Mapping[str, str] | None = None,
    force_format: str | None = None,
    levers: Mapping[str, object],
    profile,
    module_token_budget: int = 32768,
    max_rows_per_expert: int = 2048,
    eval_rows_per_expert: int = 128,
    cache_dir: str | Path | None = None,
    progress: bool = True,
    max_layers: int | None = None,
    render_mode: str = "batched",
    gate_calib_ids: torch.Tensor | None = None,
    gate_token_budget: int | None = None,
    module_acts_override: Mapping[str, torch.Tensor] | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    cb_serialization_context=None,
) -> dict:
    """Render packed-MoE experts into ``ProductionWeightCache`` entries.

    ``render_mode``:
      * ``"batched"`` (default, production): vectorize the GPTQ column update
        across all experts of a layer via ``gptq_obs_rounding_nvfp4_batched``
        — ~E× fewer Python iterations than per-expert. Fixed damp, no JSO/
        act-order (those force a per-expert loop that is ~hours on a 35B).
      * ``"per_expert"``: render each expert through the full
        ``render_production_weight`` stack (progressive gates + damp-sweep +
        JSO + do-no-harm). Faithful to the dense path but ~hours on a 35B;
        used for the fidelity A/B that justifies the production default.

    ``gate_calib_ids``: optional token ids from a corpus DISJOINT from
    ``calib_ids``. When given, the per-expert GPTQ-vs-RTN do-no-harm gate is
    judged on this corpus's routed rows instead of a same-corpus held-out
    slice, and GPTQ fits on ALL fit-corpus rows. Rationale (2026-06-09 served
    A/B): per-expert Hessians are thin (sparse routing), so GPTQ overfits its
    calibration *domain* — a same-domain holdout catches in-sample overfit but
    passed renders that LOST on served cross-domain KL (RTN 0.0302 vs GPTQ
    0.0334 confident). Experts the gate corpus never routes to fall back to
    the same-domain holdout. Default ``None`` preserves prior behavior.

    This closes the packed-expert RTN-by-omission gap: every non-BF16 packed
    expert tensor receives a deliberate cache render keyed by routed
    per-expert activations. In production's default ``"batched"`` mode that
    render is fixed-damp batched GPTQ without dense JSO/act-order/damp-sweep,
    followed by the measured GPTQ-vs-RTN gate below. The non-default
    ``"per_expert"`` mode is the one that calls ``render_production_weight``
    and matches the dense GPTQ+damp-sweep+JSO stack. Export then reads the
    cached 3-D dequant and re-packs it — exactly the
    ``_pack_production_cached_2d`` contract, lifted to packed experts.

    Args:
      cache: the ``ProductionWeightCache`` produced by
        ``fill_production_weight_cache``.  New 3-D ``(experts_qname.pn, fmt)``
        entries are added in place (and streamed to ``cache_dir`` shards when
        disk-streaming is on).
      render_assignment: the concrete export assignment (recipe_key -> fmt).
        Only non-BF16 packed-expert tensors are rendered (BF16 = passthrough).
      levers: the resolved production lever dict. It is consumed by
        ``render_mode="per_expert"``; the production ``"batched"`` path uses
        its fixed fast recipe regardless of dense-linears JSO/damp-sweep
        levers.
      module_token_budget: reservoir size for each experts module's input X.
      max_rows_per_expert: cap on routed rows fed to each expert's GPTQ.
      max_layers: debug/timing cap — render only the first N experts modules.
      col_weights: optional ``{qname: vector}`` imatrix map (re-vet R3 / CB
        Milestone C), keyed by the packed tensor name (``…experts.w1`` etc.)
        with either one shared ``(in_features,)`` vector or a per-expert
        ``(E, in_features)`` stack — the layout
        ``moe_imatrix.synthesize_packed_expert_col_weights`` emits. Applied
        only on the per-expert render path and only for the weighted-render
        families; NVFP4/FP8 expert bytes are unchanged.
      cb_serialization_context: explicit artifact serialization identity for
        CB expert renders. Required for CB and never inferred from process
        environment defaults.
      module_acts_override: streaming build. ``{experts_qname: X}`` module-level
        input snapshots sourced from the probe's activation cache instead of a
        fresh forward pass. When supplied, no calibration forward runs, in-scope
        experts are restricted to the supplied modules (the currently-installed
        decoder layer), ``calib_ids`` may be ``None``, and the cross-domain gate
        corpus is unsupported.

    Returns a coverage dict consumed by the export enforcement gate:
      ``{experts_qname.pn: {"fmt", "n_experts", "min_rows", "median_rows",
      "rendered", "had_activations"}}``.
    """
    import time as _time
    from prismaquant.sensitivity_probe import (
        _is_packed_experts_module,
        _packed_experts_param_names,
    )
    from prismaquant.measure_quant_cost import (
        derive_per_expert_activations,
        _packed_experts_parent_module,
    )
    from prismaquant.export_native_compressed import (
        _split_packed_expert_tensor,
        compute_nvfp4_global_real,
    )
    from prismaquant import format_registry as fr

    device = next(model.parameters()).device
    cache_dir_path = Path(cache_dir) if cache_dir is not None else None
    if cache_dir_path is not None:
        cache_dir_path.mkdir(parents=True, exist_ok=True)

    def _canon(fmt: str) -> str:
        return fr.canonical_format_name(str(fmt).strip().upper())

    if force_format is None and render_assignment is None:
        raise ValueError(
            "fill_packed_expert_cache_entries requires render_assignment or "
            "force_format")
    if force_format is not None and render_assignment is not None:
        raise ValueError(
            "fill_packed_expert_cache_entries: pass render_assignment OR "
            "force_format, not both")
    if module_acts_override is None and calib_ids is None:
        raise ValueError(
            "fill_packed_expert_cache_entries requires calib_ids unless "
            "module_acts_override supplies the experts-module input snapshot")
    if module_acts_override is not None and gate_calib_ids is not None:
        raise ValueError(
            "fill_packed_expert_cache_entries: module_acts_override (streaming) "
            "does not support a cross-domain gate corpus")

    # 1. Resolve in-scope packed-expert tensors (non-BF16 in the assignment).
    #    Each entry: (experts_qname, mod, parent, pn, full, fmt).
    in_scope: list[tuple[str, nn.Module, nn.Module, str, str, str]] = []
    experts_qnames: set[str] = set()
    modules_seen = 0
    for experts_qname, mod in model.named_modules():
        if not _is_packed_experts_module(mod, profile):
            continue
        # Streaming build: only the currently-installed decoder layer's experts
        # module is materialized (others are on meta) and only it has an X in
        # the override. Skip everything else so we never touch a meta param.
        if (
            module_acts_override is not None
            and experts_qname not in module_acts_override
        ):
            continue
        if max_layers is not None and modules_seen >= max_layers:
            break
        modules_seen += 1
        parent = _packed_experts_parent_module(model, experts_qname)
        for pn in _packed_experts_param_names(mod, profile):
            full = f"{experts_qname}.{pn}" if experts_qname else pn
            if force_format is not None:
                # Force-format mode: render every packed expert at one format,
                # ignoring render_assignment. Used by the format-menu frontier
                # build (eager NVFP4) and by per-Pareto-point lazy FP8 gap-fill.
                fmt = _canon(force_format)
            else:
                try:
                    recipe_key = profile.live_to_recipe_name(full)
                except Exception:
                    recipe_key = full
                fmt = render_assignment.get(recipe_key)
                if fmt is None and recipe_key != full:
                    fmt = render_assignment.get(full)
                if fmt is None:
                    continue
                fmt = _canon(fmt)
            if fmt == "BF16":
                continue
            in_scope.append((experts_qname, mod, parent, pn, full, fmt))
            experts_qnames.add(experts_qname)

    coverage: dict[str, dict[str, object]] = {}
    if not in_scope:
        if progress:
            print("[prod-cache/experts] no non-BF16 packed experts in scope",
                  flush=True)
        return coverage

    packed_formats_by_qname = {
        full: (fmt,)
        for _experts_qname, _mod, _parent, _pn, full, fmt in in_scope
    }
    resolved_packed_levers = _resolve_production_render_levers(levers)
    packed_mechanism_plan = _resolve_render_mechanism_plan(
        resolved_packed_levers
    )
    _extend_production_cache_cb_render_identity(
        cache,
        packed_formats_by_qname,
        cb_serialization_context=cb_serialization_context,
        col_weights=col_weights,
        render_levers=resolved_packed_levers,
        render_mechanism_plan=packed_mechanism_plan,
    )
    packed_cb_source_weights = {
        full: getattr(mod, pn).detach()
        for _experts_qname, mod, _parent, pn, full, fmt in in_scope
        if _is_cb_format_name(fmt)
    }
    if packed_cb_source_weights:
        bind_production_cache_cb_source_weights(
            cache,
            packed_cb_source_weights,
            require_complete=False,
            where="ProductionWeightCache packed source binding",
        )

    stale_cb_shards: list[Path] = []
    for _experts_qname, _mod, _parent, _pn, full, fmt in in_scope:
        if not _is_cb_format_name(fmt):
            continue
        key = (full, fmt)
        value = cache.weights.get(key)
        if value is not None and not isinstance(value, torch.Tensor):
            path = Path(cache._path_for_value(value))
            if path.is_file():
                stale_cb_shards.append(path)
        if cache_dir_path is not None:
            path = cache_dir_path / _cache_weight_filename(full, fmt)
            if path.is_file() and path not in stale_cb_shards:
                stale_cb_shards.append(path)
    if stale_cb_shards:
        raise RuntimeError(
            "production packed-CB cache resume is disabled because existing "
            "weight shards are not independently bound to the fresh "
            "serialization/imatrix identity; move or rebuild the cache "
            f"directory first. sample={stale_cb_shards[:8]}"
        )

    if progress:
        print(
            f"[prod-cache/experts] {len(in_scope)} packed-expert tensors across "
            f"{len(experts_qnames)} modules; capturing module activations "
            f"(budget={module_token_budget} tokens/module)",
            flush=True,
        )

    # RESUME: persist per-param activation max_abs (the export's W4A4 input
    # scale) to a sidecar so a resumed build — which skips re-rendering existing
    # shards — still has the calibrated scale. Without this, a resumed build
    # ships the 1.0 placeholder (Codex blocker). Load it up front; the render
    # loop writes it as each tensor is (re)computed.
    import json as _json
    if cache.activation_max_abs is None:
        cache.activation_max_abs = {}
    expert_sidecar_path: Path | None = (
        cache_dir_path / "packed_expert_max_abs.json"
        if cache_dir_path is not None else None
    )
    if expert_sidecar_path is not None and expert_sidecar_path.is_file():
        try:
            cache.activation_max_abs.update(_json.loads(
                expert_sidecar_path.read_text()))
        except Exception:
            pass

    in_scope_keys = [(full, fmt) for (_q, _m, _p, _pn, full, fmt) in in_scope]

    def _persist_expert_sidecar() -> None:
        if expert_sidecar_path is None:
            return
        # MERGE with the existing sidecar — a subset render (the M4 lazy
        # per-Pareto-point FP8 gap-fill) must not destroy the eager build's
        # scale entries for every OTHER expert in the shared cache_dir.
        merged: dict[str, float] = {}
        if expert_sidecar_path.is_file():
            try:
                merged.update(_json.loads(expert_sidecar_path.read_text()))
            except Exception:
                pass
        for full_k, _fmt in in_scope_keys:
            v = cache.activation_max_abs.get(full_k)
            if v is not None:
                merged[full_k] = float(v)
        tmp = expert_sidecar_path.with_suffix(".json.tmp")
        tmp.write_text(_json.dumps(merged, indent=2))
        os.replace(tmp, expert_sidecar_path)
    # Only capture activations if some tensor still needs rendering OR is
    # missing its calibrated scale — a fully-resumed build with a complete
    # sidecar skips the forward entirely.
    def _needs_work(full: str, fmt: str) -> bool:
        if (full, fmt) in cache.weights:
            return cache.activation_max_abs.get(full) is None
        if cache_dir_path is not None:
            shard = cache_dir_path / _cache_weight_filename(full, fmt)
            if shard.is_file():
                return cache.activation_max_abs.get(full) is None
        return True

    work_remaining = any(_needs_work(full, fmt) for full, fmt in in_scope_keys)

    module_acts: dict[str, torch.Tensor] = {}
    if module_acts_override is not None:
        # Streaming build: the experts-module input snapshot X comes from the
        # probe's activation cache (keyed by the experts-module qname), not a
        # fresh forward pass — the model is only ever one layer resident here.
        # Store on CPU/fp32 exactly as the collector would so the downstream
        # derive/render path is byte-identical to the resident build.
        module_acts = {
            q: t.detach().reshape(-1, t.size(-1)).to(
                device="cpu", dtype=torch.float32,
            )
            for q, t in module_acts_override.items()
            if q in experts_qnames
        }
        if progress:
            print(
                f"[prod-cache/experts] streaming: using {len(module_acts)} "
                "supplied module activation snapshot(s) (no forward)",
                flush=True,
            )
    elif work_remaining:
        # 2. Capture module-level X for each experts module (one calib forward).
        # Store the (large) per-module reservoirs on CPU/host, NOT in the CUDA
        # allocator pool: on the GB10 unified-memory box the model already
        # occupies ~67 GB of GPU pool, and 40 modules × budget × hidden × fp32
        # in the CUDA pool fragments/OOMs the capture (observed). CPU storage
        # keeps them out of the CUDA segment pool; derive moves one module's X
        # back to GPU transiently. Pinned for faster H2D at render.
        collector = _PackedExpertActivationCollector(
            model,
            experts_qnames,
            module_token_budget=module_token_budget,
            store_device=torch.device("cpu"),
            store_dtype=torch.float32,
            profile=profile,
        )
        collector.install()
        t_cap = _time.monotonic()
        try:
            with torch.no_grad():
                for i in range(calib_ids.size(0)):
                    batch = calib_ids[i:i + 1].to(device)
                    try:
                        model(batch, use_cache=False)
                    except TypeError:
                        model(batch)
        finally:
            collector.remove()
        module_acts = collector.collected()
        if progress:
            print(
                f"[prod-cache/experts] captured {len(module_acts)} module "
                f"activations in {_time.monotonic() - t_cap:.1f}s",
                flush=True,
            )
    elif progress:
        print(
            "[prod-cache/experts] resume: all packed experts rendered + scales "
            "in sidecar, skipping activation capture",
            flush=True,
        )

    # 2b. Cross-domain gate corpus: capture a SECOND reservoir per module from
    # the disjoint gate corpus. Same CPU-storage rationale as the fit reservoir.
    gate_module_acts: dict[str, torch.Tensor] = {}
    if work_remaining and gate_calib_ids is not None:
        gate_collector = _PackedExpertActivationCollector(
            model,
            experts_qnames,
            module_token_budget=int(gate_token_budget or module_token_budget),
            store_device=torch.device("cpu"),
            store_dtype=torch.float32,
            profile=profile,
        )
        gate_collector.install()
        t_cap = _time.monotonic()
        try:
            with torch.no_grad():
                for i in range(gate_calib_ids.size(0)):
                    batch = gate_calib_ids[i:i + 1].to(device)
                    try:
                        model(batch, use_cache=False)
                    except TypeError:
                        model(batch)
        finally:
            gate_collector.remove()
        gate_module_acts = gate_collector.collected()
        if progress:
            print(
                f"[prod-cache/experts] captured {len(gate_module_acts)} "
                f"cross-domain gate-module activations in "
                f"{_time.monotonic() - t_cap:.1f}s",
                flush=True,
            )

    # 3. Render each in-scope tensor per-expert through render_production_weight.
    derived_by_module: dict[str, dict] = {}
    gate_derived_by_module: dict[str, dict | None] = {}
    # Index of each module's LAST in-scope param, so we can free its (large,
    # GPU-resident) derived per-expert activations + captured X as soon as both
    # its params (gate_up + down) are rendered — otherwise they accumulate
    # across all 40 modules (~2.5 GB/module on GPU) and OOM the box.
    last_idx_by_module: dict[str, int] = {}
    for _i, (_q, _m, _p, _pn, _full, _fmt) in enumerate(in_scope):
        last_idx_by_module[_q] = _i
    weights = cache.weights
    t0 = _time.monotonic()
    for idx, (experts_qname, mod, parent, pn, full, fmt) in enumerate(in_scope):
        key = (full, fmt)
        fname = (
            _cache_weight_filename(full, fmt)
            if cache_dir_path is not None else None
        )
        shard_exists = (key in weights) or (
            fname is not None and (cache_dir_path / fname).is_file()
        )
        need_scale = cache.activation_max_abs.get(full) is None
        # Fully done (shard + calibrated scale): just register the path.
        if shard_exists and not need_scale:
            if key not in weights and fname is not None:
                weights[key] = fname
            continue

        X = module_acts.get(experts_qname)
        if X is None or X.numel() == 0:
            raise RuntimeError(
                f"[prod-cache/experts] no captured activations for "
                f"{experts_qname}; cannot render {full} through the deliberate "
                f"path (would silently fall back to RTN). Increase the calib "
                f"set or module_token_budget."
            )
        mod_dtype = getattr(mod, pn).dtype
        if experts_qname not in derived_by_module:
            # The router forward inside derive_per_expert_activations runs in
            # the model's compute dtype (bf16); X was stored as fp32 for the
            # reservoir, so cast it back to the experts module dtype before
            # routing. The per-expert activations are re-promoted to fp32 by
            # render_production_weight's GPTQ path.
            derived_by_module[experts_qname] = derive_per_expert_activations(
                mod, X.to(device=device, dtype=mod_dtype), parent,
                capture_down=True,
                max_rows_per_expert=max_rows_per_expert,
            )
        derived = derived_by_module[experts_qname]
        if gate_module_acts and experts_qname not in gate_derived_by_module:
            # Cross-domain gate rows: same routing derivation, capped at the
            # eval budget — the gate only judges, it never fits.
            Xg = gate_module_acts.get(experts_qname)
            gate_derived_by_module[experts_qname] = (
                derive_per_expert_activations(
                    mod, Xg.to(device=device, dtype=mod_dtype), parent,
                    capture_down=True,
                    max_rows_per_expert=eval_rows_per_expert,
                )
                if Xg is not None and Xg.numel() > 0 else None
            )
        gate_derived = gate_derived_by_module.get(experts_qname)
        packed_param = getattr(mod, pn).detach().float()  # [E, out, in]
        E = packed_param.shape[0]
        proj_split = _split_packed_expert_tensor(packed_param, pn, profile)
        is_down = len(proj_split) == 1
        acts_key = "down" if is_down else "gate_up"
        per_expert_acts = derived[acts_key]
        row_counts = derived["row_counts"]

        # Per-param activation max_abs for the export's input_global_scale
        # (W4A4 needs the calibrated activation clip; without it experts ship
        # the 1.0 placeholder — Codex blocker). gate/up share the module input
        # X (measured uncapped over the full captured reservoir); down sees the
        # post-SwiGLU intermediate. One scale per packed param, applied to all
        # experts (the fused MoE kernel quantizes the input once).
        if is_down:
            param_max_abs = 0.0
            for e in range(E):
                a = per_expert_acts[e] if e < len(per_expert_acts) else None
                if a is not None and a.numel() > 0:
                    param_max_abs = max(
                        param_max_abs, float(a.detach().abs().max().item()))
        else:
            param_max_abs = float(X.detach().abs().max().item())
        # First calibrated scale wins per qname: activation_max_abs is keyed
        # per-param (shared across formats), so a later render of ANOTHER
        # format (M4 lazy FP8 gap-fill) must not clobber the scale the eager
        # NVFP4 rung calibrated, measured under, and will ship. A fresh build
        # (no prior scale, no sidecar) always writes.
        if param_max_abs > 0 and cache.activation_max_abs.get(full) is None:
            cache.activation_max_abs[full] = param_max_abs
            _persist_expert_sidecar()

        # Resume with a missing scale: shard already on disk, we only needed to
        # recompute + persist the calibrated scale. Don't re-render.
        if shard_exists:
            if key not in weights and fname is not None:
                weights[key] = fname
            continue

        # Per-expert global: for split gate/up the joint is max over slices so
        # gate and up share one scale (matches the export packer); for down the
        # per-expert global stands alone.
        per_expert_global: list[torch.Tensor | None] = [None] * E
        for e in range(E):
            cands = [
                compute_nvfp4_global_real(sp[e].float(), group_size=16)
                for _, sp in proj_split
            ]
            per_expert_global[e] = (
                torch.stack(cands).max() if len(cands) > 1 else cands[0]
            )

        empties = sum(
            1 for e in range(E)
            if e >= len(per_expert_acts)
            or per_expert_acts[e] is None
            or per_expert_acts[e].numel() == 0
        )

        rtn_fallbacks = 0
        heldout_reverts = 0
        cross_gated = 0
        use_batched = (render_mode == "batched" and fmt == "NVFP4")
        if use_batched:
            from prismaquant.export_batched_gptq import (
                gptq_obs_rounding_nvfp4_batched,
            )
            from prismaquant.export_native_compressed import _rtn_dequant_nvfp4
            src = packed_param.to(device=device, dtype=torch.float32)
            hidden_in = packed_param.shape[2]
            # Split each expert's routed rows into a RENDER set (fit GPTQ) and a
            # HELD-OUT eval set (decide GPTQ-vs-RTN). The held-out gate makes the
            # render provably non-regressive vs RTN on data the expert was not
            # fit to — catching the rank-deficient-Hessian overfit where
            # in-sample GPTQ always "wins" but generalizes worse than RTN
            # (observed on down_proj at low row counts). When a cross-domain
            # gate corpus is available for this expert, the eval set comes from
            # THAT corpus (and GPTQ fits on all fit-corpus rows) — a same-domain
            # holdout cannot catch calibration-DOMAIN overfit, which is the
            # failure the 2026-06-09 served A/B actually exhibited. Empty
            # experts (no rows) get source RTN, never the batched core's
            # all-zero dead-column output.
            per_expert_gw = derived.get("gate_weights")
            gate_acts_lists = (
                gate_derived[acts_key] if gate_derived is not None else None
            )
            gate_gw_lists = (
                gate_derived.get("gate_weights")
                if gate_derived is not None else None
            )
            render_acts: list[torch.Tensor] = []
            eval_acts: list[torch.Tensor | None] = []
            eval_gw: list[torch.Tensor | None] = []
            for e in range(E):
                a = per_expert_acts[e] if e < len(per_expert_acts) else None
                gw = (
                    per_expert_gw[e]
                    if per_expert_gw is not None and e < len(per_expert_gw)
                    else None
                )
                xa = None
                xw = None
                if gate_acts_lists is not None and e < len(gate_acts_lists):
                    cand = gate_acts_lists[e]
                    if cand is not None and cand.numel() > 0:
                        xa = cand
                        if gate_gw_lists is not None and e < len(gate_gw_lists):
                            xw = gate_gw_lists[e]
                if a is None or a.numel() == 0:
                    render_acts.append(
                        torch.zeros((0, hidden_in), device=device,
                                    dtype=torch.float32))
                    eval_acts.append(None)
                    eval_gw.append(None)
                    continue
                a = a.to(device=device, dtype=torch.float32)
                if xa is not None:
                    # Cross-domain do-no-harm: fit on ALL fit-corpus rows,
                    # judge on the disjoint gate corpus's routed rows.
                    render_acts.append(a)
                    eval_acts.append(
                        xa.to(device=device, dtype=torch.float32))
                    eval_gw.append(
                        xw.to(device=device, dtype=torch.float32).pow(2)
                        if xw is not None else None
                    )
                    cross_gated += 1
                    continue
                n = int(a.shape[0])
                n_eval = min(eval_rows_per_expert, n // 5) if n >= 20 else 0
                if n_eval > 0:
                    render_acts.append(a[:n - n_eval])
                    eval_acts.append(a[n - n_eval:])
                    # Router-weight the held-out objective: served MoE scales
                    # each token's expert output by its top_k gate weight, so
                    # the row's output error² scales by gate_weight².
                    eval_gw.append(
                        gw[n - n_eval:].to(
                            device=device, dtype=torch.float32).pow(2)
                        if gw is not None else None
                    )
                else:
                    render_acts.append(a)
                    eval_acts.append(None)  # too few rows to hold out
                    eval_gw.append(None)
            overrides = torch.stack([
                per_expert_global[e].to(device=device, dtype=torch.float32)
                for e in range(E)
            ])
            rendered = gptq_obs_rounding_nvfp4_batched(
                src, render_acts, group_size=16,
                global_real_overrides=overrides,
                static_act_order=False,
                joint_scale_opt=False,
            )  # fp32 [E,out,in]; held-out do-no-harm below, store casts to bf16
            for e in range(E):
                w_rtn = _rtn_dequant_nvfp4(
                    src[e], group_size=16, global_real_override=overrides[e],
                )
                src_a = (
                    per_expert_acts[e] if e < len(per_expert_acts) else None
                )
                if src_a is None or src_a.numel() == 0:
                    # empty expert: no activation evidence -> source RTN
                    rendered[e] = w_rtn
                    rtn_fallbacks += 1
                    continue
                ev = eval_acts[e]
                # held-out gate when available, else in-sample render set
                gate_acts = ev if ev is not None else render_acts[e]
                row_w = eval_gw[e] if ev is not None else None
                if gate_acts is None or gate_acts.numel() == 0:
                    rendered[e] = w_rtn
                    rtn_fallbacks += 1
                    continue
                s_gptq = score_render_error(
                    src[e], rendered[e], gate_acts, row_weights=row_w)
                s_rtn = score_render_error(
                    src[e], w_rtn, gate_acts, row_weights=row_w)
                if s_rtn <= s_gptq:
                    rendered[e] = w_rtn
                    rtn_fallbacks += 1
                    if ev is not None:
                        heldout_reverts += 1
        else:
            # Per-expert full-stack render (fmt != NVFP4, or the A/B mode).
            rendered = torch.empty_like(packed_param)
            stack_cw = None if col_weights is None else col_weights.get(full)
            for e in range(E):
                w_e = packed_param[e]  # [out, in]
                acts_e = (
                    per_expert_acts[e] if e < len(per_expert_acts) else None
                )
                has_acts = acts_e is not None and acts_e.numel() > 0
                w_dq = render_production_weight(
                    w_e, fmt,
                    qname=f"{full}.e{e}",
                    activations=(
                        {f"{full}.e{e}": acts_e.to(device)} if has_acts else {}
                    ),
                    levers=levers,
                    joint_global_real=per_expert_global[e],
                    col_weights=_expert_col_weights(stack_cw, e, int(E)),
                    cb_serialization_context=cb_serialization_context,
                )
                rendered[e] = w_dq.to(rendered.dtype)
                del w_dq
            rendered = rendered.to(getattr(mod, pn).dtype)

        _store_rendered_weight_entry(
            weights=weights,
            cache_dir_path=cache_dir_path,
            qname=full,
            fmt=fmt,
            tensor=rendered,
            weight_dtype=getattr(mod, pn).dtype,
        )
        pos_rows = sorted(r for r in row_counts if r > 0)
        coverage[full] = {
            "fmt": fmt,
            "n_experts": int(E),
            "min_rows": int(pos_rows[0]) if pos_rows else 0,
            "median_rows": int(pos_rows[len(pos_rows) // 2]) if pos_rows else 0,
            "empty_experts": int(empties),
            "rtn_fallbacks": int(rtn_fallbacks),
            "heldout_reverts": int(heldout_reverts),
            "gptq_experts": int(E - rtn_fallbacks),
            "render_mode": render_mode,
            "gate_mode": (
                "cross-domain" if (use_batched and gate_derived is not None)
                else "in-domain-holdout"
            ),
            "cross_gated_experts": int(cross_gated),
            "rendered": True,
            "had_activations": True,
        }
        del rendered, packed_param, proj_split
        # Free this module's GPU-resident derived activations + captured X once
        # its last param is rendered (bounds peak to ~one module's working set).
        if idx == last_idx_by_module.get(experts_qname):
            derived = None
            gate_derived = None
            derived_by_module.pop(experts_qname, None)
            module_acts.pop(experts_qname, None)
            gate_derived_by_module.pop(experts_qname, None)
            gate_module_acts.pop(experts_qname, None)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if progress:
            print(
                f"[prod-cache/experts] {idx + 1}/{len(in_scope)} {full} @ {fmt} "
                f"E={E} empty={empties} rtn_fallback={rtn_fallbacks} "
                f"(heldout_reverts={heldout_reverts}, "
                f"cross_gated={cross_gated}) gptq={E - rtn_fallbacks} "
                f"min_rows={coverage[full]['min_rows']} "
                f"({_time.monotonic() - t0:.1f}s elapsed)",
                flush=True,
            )

    if progress:
        print(
            f"[prod-cache/experts] rendered {len(coverage)} packed-expert "
            f"tensors in {_time.monotonic() - t0:.1f}s",
            flush=True,
        )
    return coverage
