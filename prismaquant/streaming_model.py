#!/usr/bin/env python3
"""streaming_model.py — shared streaming-skeleton infrastructure.

Factored out of `incremental_probe.py` so the cost-measurement side
(`incremental_measure_quant_cost.py`) can reuse the exact same
"skeleton-on-meta, head-resident, decoder-layers-swap" plumbing without
copy-pasting.

What lives here:

  - `StreamingContext`: holds the model, per-layer install resolvers,
    weight map, LayerCache, and a single-worker prefetch pool. Built once,
    reused across every shard.
  - `_build_streaming_context`: one-time setup (AutoConfig, empty
    skeleton, `from_pretrained` with explicit device_map pinning head
    resident and decoder layers to disk, strip accelerate hooks, unload
    layers back to meta).
  - `_classify_shard`: maps a shard-include regex to one of
    {"body", "mtp", "visual", "lm_head"}.

What stays in `incremental_probe`:
  - `build_layer_shard_regexes` / `build_extended_shard_regexes`
  - `load_num_hidden_layers`
  - Body/MTP shard runners (those are Fisher-semantics-specific).

The cost side will import from both this module and
`incremental_probe` (for the regex builders) — the regex helpers are
stable public API that both sides share.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from safetensors import safe_open

try:
    from accelerate import init_empty_weights
except ModuleNotFoundError:
    @contextmanager
    def init_empty_weights():
        with torch.device("meta"):
            yield

try:
    from accelerate.hooks import remove_hook_from_module
except ModuleNotFoundError:
    def remove_hook_from_module(module, recurse: bool = False):
        del recurse
        return module

from .autoscale import (
    _PACKED_BYTE_DTYPES,
    declared_expert_dtype_covers,
    declared_fp4_expert_dtype,
)
from .layer_streaming import (
    _build_fp8_scale_inv_map,
    LayerCache,
    _build_concat_merger,
    _build_expert_packer,
    _build_install_resolver,
    _build_weight_map,
    _fast_install,
    _get_layer_list,
    _get_rotary,
    _head_prefixes,
    _materialize,
    _read_layer_to_device,
    _resolve_base_prefix,
    _unload,
    set_module_tensor_to_device,
)
from .tied_embeddings import resolve_tied_output_embedding
from .safetensors_pread import (
    PREAD_BACKEND,
    read_safetensors_metadata,
    resolve_safetensors_backend,
)


def _bypass_hf_fp8_module_rewrite(model_path: str) -> bool:
    """True when HF's FP8 pre-load module rewrite must be skipped here.

    Two independent conditions, and they belong in different places:

      - the **checkpoint** is native FP8 with block scales — a per-checkpoint
        fact, read from `quantization_config` right here;
      - the **architecture**'s expert container breaks under that rewrite — a
        static architecture property, so it is declared in the model profile
        (`staging.bypass_hf_fp8_module_rewrite`) rather than pattern-matched
        on the model name. MiniMax-M2/M2.7 exposes 256 experts as a
        `ModuleList`; transformers 5.x replaces it with `FP8Experts` and then
        tries to set `experts.0.w1`, which that container does not support.

    The streaming path never needs the rewrite anyway: `_read_layer_to_device`
    reads the source fp8 bytes and applies `.weight_scale_inv` inline.
    """
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        return False
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "fp8" or "weight_block_size" not in qc:
        return False
    try:
        from .model_profiles import profile_from_config

        return bool(profile_from_config(cfg).bypass_hf_fp8_module_rewrite())
    except Exception:
        return False


@contextmanager
def _mask_cuda_queries_during_meta_init(log_prefix: str):
    """Keep HF meta-skeleton construction from probing CUDA.

    `init_empty_weights()` should be a pure Python/meta-tensor path. Some model
    constructors or optional attention backends still ask `torch.cuda` whether
    a device exists while choosing implementation details. On systems with a
    wedged or slow UVM/NVML path that can burn CPU or hang before PrismaQuant
    reaches its own streaming loader. The skeleton does not need CUDA, so make
    those availability checks return "no CUDA" for this short block without
    initializing CUDA or changing the requested runtime device.
    """
    enabled = os.environ.get(
        "PRISMAQUANT_MASK_CUDA_DURING_META_INIT", "1"
    ).lower() not in {"0", "false", "no"}
    if not enabled or torch.cuda.is_initialized():
        yield
        return

    # Prime transformers' lru_cached fla / causal-conv1d availability checks
    # with CUDA visible BEFORE masking it. Several modeling files
    # (Qwen3.5/3.6 MoE, Qwen3-Next, OLMo-hybrid) bind their gated-delta-rule
    # FAST PATH at *module import time* behind
    # `if is_flash_linear_attention_available():`. That check is
    # `@lru_cache`d and CUDA-gated, so if the module is first imported inside
    # this mask it caches `False`, the fla ops are never imported, and the
    # fast path is silently lost for the whole process — falling back to the
    # slow torch gated-delta-rule path (issue #4). Re-priming the caches here
    # pins them to the real CUDA state so the subsequent masked import still
    # binds the fast path. No-op when the packages aren't installed; the
    # availability call is a lightweight `torch.cuda.is_available()` (set
    # PRISMAQUANT_MASK_CUDA_DURING_META_INIT=0 to skip the mask entirely on a
    # pathologically-wedged UVM where even that probe is slow).
    try:
        from transformers.utils import import_utils as _tiu
        for _avail in ("is_flash_linear_attention_available",
                       "is_causal_conv1d_available"):
            _f = getattr(_tiu, _avail, None)
            if _f is None:
                continue
            if hasattr(_f, "cache_clear"):
                _f.cache_clear()
            _f()  # prime with CUDA visible (result cached for the process)
    except Exception:
        pass

    # Pin fla's OWN module-level device detection too. fla.utils computes
    # `device`/`device_platform`/`device_torch_lib` from triton's driver target
    # at import; if that first import lands inside this mask it caches CPU and
    # the gated-delta-rule autocast wrapper later crashes on `torch.cpu.device`
    # (the linear-attn fast path on Qwen3.5/3.6). Import it here, CUDA visible,
    # so the detection caches the real device for the process. Needs a writable
    # TRITON_CACHE_DIR — /home/rob/.triton/cache can be root-owned from docker
    # runs, which makes triton's compile cache unwritable and silently rolls fla
    # back to CPU; point TRITON_CACHE_DIR at a user-owned dir if so.
    try:
        import fla.utils  # noqa: F401  (device detection cached at import)
    except Exception:
        pass

    old_is_available = torch.cuda.is_available
    old_device_count = torch.cuda.device_count
    old_current_device = torch.cuda.current_device
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    torch.cuda.device_count = lambda: 0  # type: ignore[assignment]
    torch.cuda.current_device = lambda: 0  # type: ignore[assignment]
    try:
        print(f"{log_prefix} masking torch.cuda queries during meta init",
              flush=True)
        yield
    finally:
        torch.cuda.is_available = old_is_available  # type: ignore[assignment]
        torch.cuda.device_count = old_device_count  # type: ignore[assignment]
        torch.cuda.current_device = old_current_device  # type: ignore[assignment]


def _init_rotary_inplace(base_model: nn.Module, device: torch.device,
                         dtype: torch.dtype) -> None:
    """Populate deterministic rotary buffers on a meta-built skeleton.

    Most architectures expose one set of rotary parameters keyed by
    `inv_freq` / `original_inv_freq` — handled by the default branch.
    Architectures with multi-layer-type rotaries (DSv4, Gemma3) override
    via `ModelProfile.init_rotaries(...)` to register `<name>_inv_freq`
    buffers per layer-type (refactor #32).
    """
    rotary = _get_rotary(base_model)
    if rotary is None:
        return
    cfg = getattr(rotary, "config", None)
    if cfg is None:
        return
    try:
        rope_init_fn = rotary.compute_default_rope_parameters
    except AttributeError:
        return

    # Profile-driven dispatch first. If the profile fully handled rotary
    # init (DSv4 multi-layer-type pattern), exit. Otherwise fall through.
    try:
        from .model_profiles import profile_from_model
        if profile_from_model(base_model).init_rotaries(
                rotary, cfg, device, dtype, base_model=base_model):
            return
    except Exception:
        # Defensive: fall through to default if profile dispatch breaks.
        pass

    if hasattr(rotary, "reset_rope_cache"):
        rotary.reset_rope_cache(device)
        return

    # Single-rope path (the common case for Qwen / MiniMax / DSv3).
    inv_freq, attention_scaling = rope_init_fn(cfg, device)
    rotary.register_buffer("inv_freq", inv_freq.to(
        dtype=torch.float32, device=device), persistent=False)
    if hasattr(rotary, "original_inv_freq"):
        rotary.register_buffer(
            "original_inv_freq",
            inv_freq.to(dtype=torch.float32, device=device).clone(),
            persistent=False,
        )
    rotary.attention_scaling = attention_scaling


def _safetensors_cache_dtype_bytes(dtype_name: str,
                                   target_dtype: torch.dtype,
                                   *, fp4_packed: bool = False) -> int:
    """Bytes a safetensors tensor will occupy in the layer cache,
    per *on-disk* element (safetensors counts packed bytes as elements)."""
    dtype_name = str(dtype_name).upper()
    # Floating checkpoint tensors are cast to the requested execution
    # dtype by `_read_layer_to_device` before caching. Native FP8 source
    # weights therefore cache as bf16/fp16/fp32 after block dequant.
    if dtype_name.startswith("F") or dtype_name == "BF16":
        return torch.empty((), dtype=target_dtype).element_size()
    if fp4_packed and dtype_name in _PACKED_BYTE_DTYPES:
        # Declared MXFP4 nibble-pack (DSv4-Flash routed experts): one
        # packed I8/U8 byte dequants to TWO logical elements of the
        # execution dtype (4 bytes/disk-byte at bf16). Sizing it as the
        # verbatim 1 byte undercounts the resident tensor 4x, so
        # prepare_for_load() under-evicts and prefetch refuses layers
        # that would actually fit.
        return 2 * torch.empty((), dtype=target_dtype).element_size()
    return {
        "BOOL": 1,
        "U8": 1, "I8": 1,
        "U16": 2, "I16": 2,
        "U32": 4, "I32": 4,
        "U64": 8, "I64": 8,
    }.get(dtype_name, 1)


def _estimate_layer_cache_bytes(
    *,
    weight_shard: dict[str, str],
    weight_ckpt: dict[str, str],
    layers_prefix: str,
    num_layers: int,
    target_dtype: torch.dtype,
    fp4_experts: bool = False,
    safetensors_backend: str | None = None,
) -> tuple[int, list[int]]:
    """Estimate dequanted cache bytes per decoder layer without loading data.

    `fp4_experts` is the checkpoint's explicit packed-FP4 expert
    declaration (`declared_fp4_expert_dtype`): expert I8/U8 tensors —
    routed and shared alike, see `declared_expert_dtype_covers` — then
    price as MXFP4 nibble-packs (2 logical elements/byte at the execution
    dtype) instead of verbatim int8."""
    pat = re.compile(rf"^{re.escape(layers_prefix)}(?P<idx>\d+)\.")
    by_shard: dict[str, list[tuple[int, str, bool]]] = {}
    for model_name, shard in weight_shard.items():
        m = pat.match(model_name)
        if m is None:
            continue
        idx = int(m.group("idx"))
        if idx < 0 or idx >= num_layers:
            continue
        by_shard.setdefault(shard, []).append((
            idx, weight_ckpt[model_name],
            bool(fp4_experts and declared_expert_dtype_covers(model_name)),
        ))

    backend = resolve_safetensors_backend(safetensors_backend)
    sizes = [0 for _ in range(num_layers)]
    try:
        for shard, pairs in by_shard.items():
            if backend == PREAD_BACKEND:
                metadata = read_safetensors_metadata(shard)
                for idx, ckpt_name, fp4_packed in pairs:
                    info = metadata[ckpt_name]
                    n = math.prod(info.shape)
                    sizes[idx] += n * _safetensors_cache_dtype_bytes(
                        info.dtype, target_dtype,
                        fp4_packed=fp4_packed)
            else:
                with safe_open(shard, framework="pt") as f:
                    for idx, ckpt_name, fp4_packed in pairs:
                        sl = f.get_slice(ckpt_name)
                        n = math.prod(int(dim) for dim in sl.get_shape())
                        sizes[idx] += n * _safetensors_cache_dtype_bytes(
                            sl.get_dtype(), target_dtype,
                            fp4_packed=fp4_packed)
    except Exception:
        if backend == PREAD_BACKEND:
            # The explicit mmap-free backend is also the validation contract:
            # corruption or a missing key must not silently turn into a zero
            # estimate and an unsafe cache/prefetch budget.
            raise
        return 0, sizes
    nonzero = [s for s in sizes if s > 0]
    return (max(nonzero) if nonzero else 0), sizes


def _auto_prefetch_workers(cache_bytes: int, layer_bytes: int,
                           requested: Any = None) -> tuple[int, str]:
    raw = requested
    if raw is None:
        raw = os.environ.get("PREFETCH_WORKERS", "auto")
    if str(raw).strip().lower() not in ("", "auto"):
        return max(1, int(raw)), "explicit"
    if layer_bytes <= 0:
        return 3, "auto-fallback"
    cache_slots = max(1, int(cache_bytes // layer_bytes))
    # Each active worker can hold one not-yet-cached layer in addition to
    # the cache itself. Bound concurrency by cache slots so prefetch does
    # not double memory pressure on small-memory runs.
    workers = min(4, max(1, cache_slots))
    return workers, "auto"


def _auto_prefetch_min_available_bytes(layer_bytes: int,
                                       requested: Any = None) -> tuple[int, str]:
    raw = requested
    if raw is None:
        raw = os.environ.get("PREFETCH_MIN_AVAILABLE_GB", "auto")
    if str(raw).strip().lower() not in ("", "auto"):
        return int(float(raw) * 1024 ** 3), "explicit"
    # Keep enough slack for at least two full dequanted layers plus a
    # fixed floor. On UMA systems this guards both CPU RAM and CUDA
    # allocations, since they share the same physical memory.
    floor = 8 * 1024 ** 3
    if layer_bytes <= 0:
        return floor, "auto-fallback"
    return max(floor, int(2 * layer_bytes)), "auto"


# ---------------------------------------------------------------------------
# Shard classification. Each shard regex falls into exactly one of these
# kinds and is orchestrated by the matching runner in the probe / cost
# script. "body" and "mtp" are the active paths; "visual" is acknowledged
# but skipped in the text-only streaming pipeline.
# ---------------------------------------------------------------------------
_BODY_SHARD_RE = re.compile(r"^model\\\.layers\\\.")
_MTP_SHARD_RE = re.compile(r"mtp\\\.(?:fc|layers\\\.)")
_VISUAL_SHARD_RE = re.compile(r"^model\\\.visual\\\.")
_LM_HEAD_SHARD_RE = re.compile(r"^\^lm_head\$?$")


def _classify_shard(regex: str) -> str:
    if _BODY_SHARD_RE.match(regex):
        return "body"
    if _MTP_SHARD_RE.search(regex):
        return "mtp"
    if _VISUAL_SHARD_RE.match(regex):
        return "visual"
    if _LM_HEAD_SHARD_RE.match(regex):
        return "lm_head"
    return "body"  # conservative fallback: treat as a body pattern


# ---------------------------------------------------------------------------
# Streaming context: skeleton + head resident + per-layer resolvers + cache.
# Built once for the whole run and reused across every shard. Holding this
# object idle between shards costs the head weights + cache RAM only;
# decoder layers live on meta or on disk and get installed transiently.
def _prefetch_delivery_enabled() -> bool:
    """Prefetch delivery: a completed speculative read is always handed to
    its consumer, even when the cache declines to retain it.

    ``PRISMAQUANT_PREFETCH_DELIVERY=0`` reproduces the whole pre-2026-08-26
    loader schedule for a controlled A/B — discard-on-refusal, no admission
    bound, no walk top-up, static-budget lookahead. Production leaves it on.
    """
    return str(
        os.environ.get("PRISMAQUANT_PREFETCH_DELIVERY", "1")
    ).strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
class StreamingContext:
    # Compatibility default for lightweight test/diagnostic contexts built
    # with ``object.__new__`` rather than the production constructor.
    safetensors_backend = "safe_open"

    def __init__(self, *, model, base_model, layers, layers_prefix: str,
                 num_layers: int, install_resolvers: list[dict],
                 weight_shard: dict[str, str], weight_ckpt: dict[str, str],
                 layer_cache: LayerCache, prefetch_pool: ThreadPoolExecutor,
                 device: torch.device, dtype: torch.dtype, offload_folder: str,
                 visual_module: Any | None = None,
                 visual_prefix: str | None = None,
                 multimodal: bool = False,
                 fp8_scale_inv_map: dict[str, tuple[str, str]] | None = None,
                 estimated_layer_bytes: int = 0,
                 prefetch_workers: int = 3,
                 prefetch_min_available_bytes: int = 0,
                 safetensors_backend: str = "safe_open",
                 expert_packer=None,
                 concat_merger=None):
        self.model = model
        self.base_model = base_model
        self.layers = layers
        self.layers_prefix = layers_prefix
        self.num_layers = num_layers
        self.install_resolvers = install_resolvers
        self.weight_shard = weight_shard
        self.weight_ckpt = weight_ckpt
        self.layer_cache = layer_cache
        self.prefetch_pool = prefetch_pool
        self.device = device
        self.dtype = dtype
        self.offload_folder = offload_folder
        # Populated when `_build_streaming_context(..., multimodal=True)`:
        # full visual tower resident on `device`, requires_grad=True on
        # Linear params so Fisher hooks fire in run_multimodal_visual_probe_pass.
        # Also exposes `visual_prefix` so cost / probe code can iterate
        # over visual Linears under `model.visual.*` (or whatever the
        # declared multimodal arch calls it).
        self.visual_module = visual_module
        self.visual_prefix = visual_prefix
        self.multimodal = multimodal
        self.estimated_layer_bytes = int(estimated_layer_bytes or 0)
        self.max_cache_slots = layer_cache.max_entries
        self.prefetch_workers = int(prefetch_workers)
        self.prefetch_min_available_bytes = int(prefetch_min_available_bytes or 0)
        self.safetensors_backend = resolve_safetensors_backend(
            safetensors_backend
        )
        self.prefetch_memory_skips = 0
        # Native-FP8 checkpoint dequant map: `{live_weight_key:
        # (shard_path, scale_inv_ckpt_key)}`. When non-empty, every
        # per-layer reload via `_read_layer_to_device` applies the
        # 128x128 block dequant inline so `mod.weight` holds true
        # dequanted weights, not raw fp8 codes cast to bf16. Empty dict
        # for BF16-native checkpoints — loader path is unchanged.
        self.fp8_scale_inv_map = fp8_scale_inv_map or {}
        # Optional per-expert -> packed-3D bridge for checkpoints that ship
        # MoE experts unfused while the live module is packed. None for
        # every other checkpoint/model (zero behavior change). Built once
        # in `_build_streaming_context` from the model profile's spec.
        self.expert_packer = expert_packer
        # Optional N->1 concat bridge for checkpoints that store one live
        # parameter as several source tensors (transformers'
        # `Concatenate(dim=...)` merges, declared per family as the spec's
        # `concat_merges`). None for every other checkpoint/model (zero
        # behavior change). Built once in `_build_streaming_context`.
        self.concat_merger = concat_merger
        self._inflight: dict[int, Any] = {}
        self._inflight_lock = threading.Lock()
        # Sequential-walk tracking for the automatic prefetch top-up.
        # Every streamed consumer (probe phase-1/phase-3, cost_streaming's
        # forward and reverse sweeps, incremental_measure_quant_cost) walks
        # layers with a +-1 stride; `install()` infers that stride and keeps
        # the achievable lookahead window enqueued, so a caller whose own
        # `schedule_prefetch(L + depth)` call was refused under memory
        # pressure still gets the nearer layers queued on the next step
        # instead of dropping to a cold read for the rest of the walk.
        self._last_installed: int | None = None
        self._walk_step: int = 0
        # Diagnostics: reads the cache declined to retain but that were
        # still delivered to the consumer from the in-flight future. Before
        # the delivery fix these were silently discarded and re-read.
        self.prefetch_delivered_unretained = 0
        self.prefetch_released_stale = 0
        self.configure_runtime_pressure_floor()

    def memory_pressure_floor_bytes(self) -> int:
        """Available-memory floor used for speculative loads and pressure trims."""
        return max(
            int(self.prefetch_min_available_bytes or 0),
            int(self.layer_cache.dynamic_reserve_bytes or 0),
        )

    def configure_runtime_pressure_floor(self) -> int:
        floor = self.memory_pressure_floor_bytes()
        self.layer_cache.configure_pressure_threshold(floor)
        return floor

    def _prefetch_worker(self, L: int):
        # v20 fix #1: re-check memory + pre-evict before the read.
        # schedule_prefetch's check may be stale if the queue was deep,
        # and the cache's dynamic budget only kicks in at put() time —
        # which is too late on UMA where the read itself can OOM.
        #
        # Cancelling here — BEFORE the read — is the only legal way to
        # decline a scheduled prefetch. Once the bytes are on the wire the
        # consumer is going to demand them within `lookahead` steps and
        # will force-insert them itself, so discarding a finished read
        # saves zero peak memory and costs one full re-read.
        pressure_floor = self.memory_pressure_floor_bytes()
        if pressure_floor > 0:
            try:
                import psutil
                if psutil.virtual_memory().available < pressure_floor:
                    self.prefetch_memory_skips += 1
                    with self._inflight_lock:
                        self._inflight.pop(L, None)
                    return None
            except Exception:
                pass
        self.layer_cache.prepare_for_load(self.estimated_layer_bytes)
        prefix = f"{self.layers_prefix}{L}."
        tensors = _read_layer_to_device(
            prefix, self.weight_shard, self.weight_ckpt, self.dtype,
            self.device, fp8_scale_inv_map=self.fp8_scale_inv_map,
            pack_experts=self.expert_packer,
            merge_concat=self.concat_merger,
            safetensors_backend=self.safetensors_backend)
        # The cache may still decline to RETAIN the layer under its dynamic
        # budget (or evict it as `pinned_until_read` before the consumer
        # arrives). That is a retention decision, not a delivery decision:
        # this future keeps the tensors reachable until `ensure_loaded`
        # claims them. The inflight entry is deliberately NOT popped here —
        # popping on completion is what previously turned every refused or
        # pre-empted put into a second synchronous read of the same bytes.
        retained = self.layer_cache.put(
            L, tensors, force=False, pinned_until_read=True)
        if not _prefetch_delivery_enabled():
            # Pre-fix reproduction lever (A/B only): pop on completion and
            # drop the tensors when the cache refused to retain them, which
            # is what forced the consumer into a second synchronous read of
            # the identical bytes. Never set this in production.
            with self._inflight_lock:
                self._inflight.pop(L, None)
            if not retained:
                return None
        return tensors

    def _claim_inflight(self, L: int):
        """Drop layer L's prefetch future, releasing its tensor reference."""
        with self._inflight_lock:
            return self._inflight.pop(L, None)

    def affordable_prefetch_slots(self) -> int:
        """How many layer-sized speculative reads host memory can carry.

        This is the admission budget: `put()`'s dynamic cap and the
        pressure floor both measure MemAvailable, so the prefetch scheduler
        has to measure it too or it enqueues reads the retention path is
        guaranteed to reject.
        """
        est = int(self.estimated_layer_bytes or 0)
        if est <= 0:
            return max(1, self.prefetch_workers)
        floor = self.memory_pressure_floor_bytes()
        try:
            import psutil
            avail = int(psutil.virtual_memory().available)
        except Exception:
            return max(1, self.prefetch_workers)
        return max(0, (avail - int(floor)) // est)

    def _release_stale_prefetches(self, keep: set[int]) -> int:
        """Drop finished prefetches the walk has moved past.

        Futures now hold their layer's bytes until claimed, so a walk that
        turns around (phase-1 forward -> phase-3 reverse) must be able to
        hand those bytes back. Only finished-or-cancellable entries are
        dropped; a read still in progress is left to complete.
        """
        released = 0
        with self._inflight_lock:
            for idx in list(self._inflight):
                if idx in keep:
                    continue
                fut = self._inflight[idx]
                if fut.cancel() or fut.done():
                    self._inflight.pop(idx, None)
                    released += 1
        self.prefetch_released_stale += released
        return released

    def schedule_prefetch(self, L: int):
        # A one-slot policy is used by the DSv4 anchored-AURA campaign on a
        # unified-memory Spark.  Speculative loading while the current layer
        # is installed would create a second live source-weight plane even if
        # the eventual cache insertion evicted back to one entry.
        if self.max_cache_slots == 1:
            return None
        if L < 0 or L >= self.num_layers:
            return None
        if self.layer_cache.peek(L):
            return None
        pressure_floor = self.memory_pressure_floor_bytes()
        if pressure_floor > 0:
            try:
                import psutil
                if psutil.virtual_memory().available < pressure_floor:
                    self.prefetch_memory_skips += 1
                    return None
            except Exception:
                pass
        with self._inflight_lock:
            if L in self._inflight:
                return self._inflight[L]
            # Admission reserves only unfinished reads. A completed future is
            # intentionally kept here until its consumer claims it, but its
            # tensor storage is already reflected in MemAvailable (and may be
            # aliased by LayerCache). Counting that delivery alias again as a
            # future allocation double-reserves the same layer and can refuse
            # a safe tail refill. At least one unfinished read is still
            # admitted above the floor: that is the overlap the walk needs,
            # and it is exactly what the cold path would allocate anyway.
            if _prefetch_delivery_enabled():
                unfinished = sum(
                    1 for fut in self._inflight.values() if not fut.done())
                if unfinished >= max(1, self.affordable_prefetch_slots()):
                    self.prefetch_memory_skips += 1
                    return None
            fut = self.prefetch_pool.submit(self._prefetch_worker, L)
            self._inflight[L] = fut
            return fut

    def _top_up_prefetch(self, L: int) -> None:
        """Keep the achievable lookahead window enqueued for a +-1 walk.

        Callers already issue one `schedule_prefetch(L + depth)` per step,
        but that single call is fire-and-forget: when it is refused (memory
        pressure, admission bound) nothing ever re-queues that layer and the
        walk falls to cold reads for the rest of the sweep. Re-deriving the
        window each step, nearest layer first, makes the schedule
        self-healing and covers every streamed consumer at once.
        """
        prev = self._last_installed
        self._last_installed = L
        step = 0 if prev is None else L - prev
        if step in (1, -1):
            self._walk_step = step
        elif step != 0:
            # Random access (polish flips, isolated re-runs): no direction
            # to extrapolate, so don't speculate.
            self._walk_step = 0
        step = self._walk_step
        if (not step or self.max_cache_slots == 1
                or not _prefetch_delivery_enabled()):
            return
        depth = min(self.suggest_prefetch_lookahead(),
                    max(1, self.affordable_prefetch_slots()))
        if depth <= 0:
            return
        window = [L + step * k for k in range(1, depth + 1)]
        self._release_stale_prefetches({L, *window})
        for target in window:
            self.schedule_prefetch(target)

    def ensure_loaded(
        self,
        L: int,
        *,
        require_prefetched: bool = False,
    ) -> tuple[dict[str, torch.Tensor], str]:
        """Return one resident layer, optionally refusing a cold source read.

        ``require_prefetched`` is the fail-closed production mode for a
        traversal that has already established its prefetch schedule.  A hot
        cache entry is accepted; an in-flight prefetch is awaited.  If neither
        produces a resident entry, fail before calling the synchronous shard
        loader so a broken schedule cannot silently turn a GPU-bound campaign
        into serialized NVMe I/O.
        """
        cached = self.layer_cache.get(L)
        if cached is not None:
            self._claim_inflight(L)
            return cached, "hot"
        with self._inflight_lock:
            fut = self._inflight.get(L)
        if fut is not None:
            delivered = fut.result()
            self._claim_inflight(L)
            cached = self.layer_cache.get(L)
            if cached is not None:
                return cached, "wait"
            if delivered:
                # The read completed but the cache declined to retain it
                # (dynamic budget refusal, or a pressure trim / pinned
                # eviction that beat the consumer here). Deliver it anyway
                # and insert on the cold path's terms — the consumer needs
                # this layer now, so re-reading the identical bytes buys
                # nothing but disk time.
                self.prefetch_delivered_unretained += 1
                self.layer_cache.put(L, delivered)
                return delivered, "wait"
        if require_prefetched:
            raise RuntimeError(
                f"streamed layer {L} is not resident after its required "
                "prefetch; refusing synchronous cold source read"
            )
        # v20 fix #1: pre-evict to make room for the synchronous read.
        # Cold path can't skip (the consumer needs this layer now), so
        # prepare_for_load best-efforts; if effective_max < layer size,
        # the cache still inserts (correctness > budget for cold).
        self.layer_cache.prepare_for_load(self.estimated_layer_bytes)
        prefix = f"{self.layers_prefix}{L}."
        tensors = _read_layer_to_device(
            prefix, self.weight_shard, self.weight_ckpt, self.dtype,
            self.device, fp8_scale_inv_map=self.fp8_scale_inv_map,
            pack_experts=self.expert_packer,
            merge_concat=self.concat_merger,
            safetensors_backend=self.safetensors_backend)
        self.layer_cache.put(L, tensors)
        return tensors, "cold"

    def install(self, L: int, *, require_prefetched: bool = False):
        tensors, src = self.ensure_loaded(
            L, require_prefetched=require_prefetched,
        )
        _fast_install(self.install_resolvers[L], tensors, self.device, model=self.model)
        # Re-derive the lookahead window now that this layer's slot is free.
        self._top_up_prefetch(L)
        # v20 step 3+4: value-aware retention. The historical
        # one-way-stream assumption (discard immediately after install)
        # is wrong for multi-shard workloads where every phase-3 shard
        # re-traverses all layers. _fast_install rebinds tensors by
        # reference, so the cache entry shares storage with the
        # model — keeping it costs no extra memory until the model
        # unload()s, and even then the entry is bounded by the cache's
        # dynamic budget (eviction follows LRU in put() when full).
        # Layers the scheduler has provably finished with are filtered
        # out via mark_done (v20 step 2).
        return src

    def unload(self, L: int):
        _unload(self.model, [f"{self.layers_prefix}{L}."])
        return self.layer_cache.trim_for_memory_pressure()

    def shutdown(self):
        self.prefetch_pool.shutdown(wait=True)
        # Completed-but-unclaimed futures hold a reference to their layer
        # tensors (that is the delivery guarantee); drop them so a torn-down
        # context does not pin layer bytes until it is garbage-collected.
        with self._inflight_lock:
            self._inflight.clear()

    def reset_between_chunks(self, retain_cache: bool = False) -> dict:
        """Drop accumulated state at chunk boundaries in the multi-chunk
        in-process probe driver. Returns memory-delta diagnostics so the
        caller can verify the cleanup actually freed memory.

        What always gets reset:
        - inflight prefetches: cancel pending futures (their results
          would be stale for the next chunk's calibration data anyway)
        - mark_done / priority / pressure-threshold config (per-shard
          state from chunk N must not leak into chunk N+1)
        - prefetch_memory_skips counter: zeroed for clean per-chunk stats
        - PyTorch CUDA caching allocator: forced release
        - Python gc: forced collection

        When `retain_cache=False` (default): the layer_cache is also
        purged. This is the v20 behavior — assumed safe and frees the
        most memory.

        When `retain_cache=True`: layer_cache contents are preserved.
        Layer weights are model-invariant across chunks (the calibration
        data changes, the model doesn't), so an entry that survived the
        end of chunk N's phase-3 reverse sweep is still byte-identical
        to what chunk N+1's phase-1 forward needs. Cuts cold-load wall
        time on the next chunk's phase-1 by the cache hit rate.

        The retention is bounded by the cache's existing dynamic budget;
        chunk N+1's first put() will evict via LRU as needed. Marker
        sets (priority / done) are still cleared so they don't poison
        chunk N+1's per-shard logic.

        Does NOT touch the loaded model itself (that's the whole point
        of the in-process driver — keep the model+offload index resident).
        """
        import gc, psutil
        before_avail = psutil.virtual_memory().available
        # Cancel inflight prefetches — they're loading layers based on
        # whatever the prior chunk's reverse sweep was scheduling, which
        # has no relevance to the next chunk's freshly-starting forward.
        with self._inflight_lock:
            for fut in self._inflight.values():
                try:
                    fut.cancel()
                except Exception:
                    pass
            self._inflight.clear()
        retained_layers = 0
        retained_bytes = 0
        if retain_cache:
            retained_layers = len(self.layer_cache._cache)
            retained_bytes = self.layer_cache.total_bytes
        else:
            # Purge the layer cache. Force-release each entry's tensors
            # before clear() so PyTorch's UMA caching allocator returns
            # the bytes (clear() alone leaves them as cache-allocator-owned).
            self.layer_cache._cache.clear()
            self.layer_cache._bytes.clear()
            self.layer_cache.total_bytes = 0
        # v20 step 2: drop the mark-done set so the next chunk's loads
        # aren't refused. Without this, layers marked done at end of
        # chunk N's phase-3 would silently fail to repopulate in chunk
        # N+1's phase-1 forward.
        self.layer_cache.clear_done()
        # v20 fix #4-A: clear priority too.
        # set_priority_layers is called per-shard from
        # _run_body_streaming_shard; carrying chunk N's priority into
        # chunk N+1 means stale layers are protected before the new
        # shard re-registers them. Reapply the pressure floor after
        # cleanup so retained caches keep responding to current memory.
        self.layer_cache.set_priority_layers(set())
        self.configure_runtime_pressure_floor()
        self.prefetch_memory_skips = 0
        self.prefetch_delivered_unretained = 0
        self.prefetch_released_stale = 0
        # The next chunk starts a fresh forward walk; a stale stride would
        # make the first install top-up in the previous chunk's direction.
        self._last_installed = None
        self._walk_step = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        after_avail = psutil.virtual_memory().available
        return {
            "before_avail_gb": before_avail / (1024 ** 3),
            "after_avail_gb": after_avail / (1024 ** 3),
            "freed_gb": (after_avail - before_avail) / (1024 ** 3),
            "retained_cache_layers": retained_layers,
            "retained_cache_gb": retained_bytes / (1024 ** 3),
        }

    def suggest_prefetch_lookahead(self) -> int:
        if self.max_cache_slots == 1:
            return 0
        if self.estimated_layer_bytes <= 0:
            if self.max_cache_slots is None:
                return 3
            return min(3, max(0, self.max_cache_slots - 1))
        cache_slots = max(
            1, int(self.layer_cache.max_bytes // self.estimated_layer_bytes))
        if self.max_cache_slots is not None:
            cache_slots = min(cache_slots, self.max_cache_slots)
        # `max_bytes` is the STATIC budget; `put()` enforces the dynamic one
        # (MemAvailable - reserve). Sizing the lookahead off the static
        # budget alone produced a depth-4 opening burst on a box that could
        # only retain one layer, so layers 1..3 were queued, refused, and
        # never re-queued. Clamp to what memory can actually carry.
        if _prefetch_delivery_enabled():
            cache_slots = min(
                cache_slots, max(1, self.affordable_prefetch_slots()) + 1)
        # Queue at most what the cache can plausibly retain. More than
        # this tends to turn prefetch into churn on memory-constrained
        # runs, especially when backward has become fast.
        # Leave one cache slot for the currently installed layer's live
        # tensors. `install()` drops cache ownership, but the model still
        # owns that layer until the caller unloads it after forward/bwd.
        if self.max_cache_slots is None:
            return max(1, min(12, cache_slots - 1))
        return max(0, min(12, cache_slots - 1))

    def prefetch_summary(self) -> str:
        with self._inflight_lock:
            inflight = len(self._inflight)
        est_gb = self.estimated_layer_bytes / (1024 ** 3)
        min_gb = self.prefetch_min_available_bytes / (1024 ** 3)
        floor_gb = self.memory_pressure_floor_bytes() / (1024 ** 3)
        from .layer_streaming import layer_read_threads
        return (f"Prefetch: workers={self.prefetch_workers} "
                f"read_threads={layer_read_threads()} "
                f"inflight={inflight} est_layer={est_gb:.1f}GB "
                f"max_cache_slots={self.max_cache_slots} "
                f"slots_affordable={self.affordable_prefetch_slots()} "
                f"min_avail={min_gb:.1f}GB "
                f"pressure_floor={floor_gb:.1f}GB "
                f"mem_skips={self.prefetch_memory_skips} "
                f"delivered_unretained={self.prefetch_delivered_unretained} "
                f"released_stale={self.prefetch_released_stale}")


def _resolve_declared_model_cls(config, default_cls):
    """Return the transformers class named by `config.architectures[0]`
    if importable, else `default_cls`. Used to bypass
    `AutoModelForCausalLM`'s silent text-only downgrade for multimodal
    umbrella configs (e.g. Qwen3_5MoeConfig → Qwen3_5MoeForCausalLM
    text-only, which drops `model.visual.*`)."""
    try:
        import transformers
        arch_names = getattr(config, "architectures", None) or []
        if arch_names and hasattr(transformers, arch_names[0]):
            return getattr(transformers, arch_names[0])
    except Exception:
        pass
    return default_cls


def _auto_causal_lm_can_resolve(config) -> bool:
    """Would `AutoModelForCausalLM.from_config(config)` find a class?

    Asks the same two questions `_BaseAutoModelClass.from_config` asks
    itself (transformers 5.6, `auto_factory.py`): is there remote code
    (`config.auto_map["AutoModelForCausalLM"]`), or is `type(config)` a
    key of `AutoModelForCausalLM._model_mapping`? Deriving the answer
    from the mapping the call itself consults — rather than from a list
    of known-wrapper class names — means anything PrismaQuant registered
    (`prismaquant/vendored`) or any config class transformers gains later
    is handled without an edit here.

    Config-only: resolves nothing and instantiates nothing.
    """
    from transformers import AutoModelForCausalLM

    auto_map = getattr(config, "auto_map", None) or {}
    if "AutoModelForCausalLM" in auto_map:
        # from_config takes its dynamic-module branch; the static mapping
        # is irrelevant there.
        return True
    try:
        # `_model_mapping` is `MODEL_FOR_CAUSAL_LM_MAPPING`; going through
        # the class attribute is what `from_config` does, so the answer
        # cannot drift from the call.
        return type(config) in AutoModelForCausalLM._model_mapping
    except Exception:
        return False


def _config_rebuilt_as(config_cls, config):
    """Rebuild `config_cls` from the TOP-LEVEL keys of a staged config.

    `stage_text_only` lifts every `text_config` key to the top level,
    drops the nested multimodal sub-configs, and rewrites
    `architectures`; after it runs, the top level *is* the text model's
    authoritative schema (that lift is exactly why `Gemma4TextConfig`
    loads correctly for the families whose profile promotes the inner
    `model_type`). The nested sub-config object still hanging off a
    wrapper config at that point is *default-constructed* and must not be
    used as a value source — verified on an Ovis2-shaped staged config,
    where `config.text_config.hidden_size` reads 4096 (the class default)
    while the checkpoint's real 64 sits at the top level.

    Sub-config keys and `model_type` are dropped (the target class owns
    its own `model_type`); derived fields such as `layer_types` are left
    out of the input so the target class recomputes them from the real
    `num_hidden_layers` instead of inheriting a default-length list.
    """
    raw = config.to_dict()
    for sub_key in set(getattr(type(config), "sub_configs", None) or {}):
        raw.pop(sub_key, None)
    raw.pop("model_type", None)
    return config_cls.from_dict(raw)


def _resolve_text_only_skeleton(config, *, log_prefix: str = "[streaming]"):
    """Return `(config, model_cls)` for a top-level config that
    `AutoModelForCausalLM` cannot resolve — i.e. a vision-language
    *wrapper* config on the TEXT-ONLY path (issue #12: MiniMax-M3's
    `MiniMaxM3VLConfig` is rejected while the accepted list in the very
    same error names `MiniMaxM3VLTextConfig`).

    Preference order, and why:

    1. **The config's own text sub-config class.** Text-only means we
       want the body and nothing else: only the `multimodal=True` branch
       of `_build_streaming_context` reads tower tensors onto the device
       (`_find_visual_module` → `_read_layer_to_device` →
       `visual_module.to(device)`), so building the declared VL
       architecture here would wire a tower of never-materialized meta
       tensors into the module tree — which the "head buffers left on
       meta" sweep in `_build_streaming_context` is right to reject —
       and would cost tower memory the moment anything did materialize
       it. The text sub-config produces exactly the module tree the
       plain-text path produces.
    2. **The declared architecture** (`config.architectures[0]`, the
       mechanism the multimodal branch already trusts), built against
       *its own* `config_class` rather than the wrapper config. This is
       the answer when staging's `ForConditionalGeneration →
       ForCausalLM` rewrite lands on a real text-only class that simply
       is not registered under the wrapper config class. Note it is
       frequently a dead end for VL wrappers, because the rewritten name
       does not exist (`Ovis2ForCausalLM`, verified absent from
       transformers 5.6) — which is the second reason it is not first.

    A registered `ModelProfile` can pre-empt all of this at staging time
    by promoting the inner `model_type`
    (`stage_text_only_promote_inner_model_type`, as Gemma 4 does), in
    which case `AutoConfig` hands back the text config class directly and
    this function is never reached.

    Raises `RuntimeError` naming every attempt when nothing resolves.
    """
    from transformers import AutoModelForCausalLM

    wrapper = type(config).__name__
    tried: list[str] = []

    # 1. text sub-config class
    text_cfg_cls = None
    try:
        sub = config.get_text_config()
        if sub is not None and sub is not config:
            text_cfg_cls = type(sub)
    except Exception as e:  # get_text_config() raises on ambiguity
        tried.append(f"{wrapper}.get_text_config() raised {e!r}")
    if text_cfg_cls is None:
        text_cfg_cls = (getattr(type(config), "sub_configs", None)
                        or {}).get("text_config")
    if text_cfg_cls is None:
        tried.append(f"{wrapper} exposes no text sub-config")
    else:
        try:
            text_cfg = _config_rebuilt_as(text_cfg_cls, config)
        except Exception as e:
            tried.append(f"rebuilding {text_cfg_cls.__name__} from the "
                         f"staged top-level config raised {e!r}")
        else:
            if _auto_causal_lm_can_resolve(text_cfg):
                print(f"{log_prefix} {wrapper} is not a CausalLM config; "
                      f"building the text-only skeleton from its text "
                      f"sub-config {text_cfg_cls.__name__} "
                      f"(no visual tower on the text-only path)",
                      flush=True)
                return text_cfg, AutoModelForCausalLM
            tried.append(f"text sub-config {text_cfg_cls.__name__} is also "
                         "absent from AutoModelForCausalLM's mapping")

    # 2. declared architecture, built against its own config class
    declared = _resolve_declared_model_cls(config, None)
    if declared is None:
        tried.append("declared architectures "
                     f"{list(getattr(config, 'architectures', None) or [])} "
                     "are not importable from transformers")
    else:
        decl_cfg_cls = getattr(declared, "config_class", None)
        if decl_cfg_cls is None or isinstance(config, decl_cfg_cls):
            decl_cfg = config
        else:
            try:
                decl_cfg = _config_rebuilt_as(decl_cfg_cls, config)
            except Exception as e:
                tried.append(f"rebuilding {decl_cfg_cls.__name__} for "
                             f"declared {declared.__name__} raised {e!r}")
                decl_cfg = None
        if decl_cfg is not None:
            print(f"{log_prefix} {wrapper} is not a CausalLM config; "
                  f"building the text-only skeleton from declared "
                  f"architecture {declared.__name__} "
                  f"(config {type(decl_cfg).__name__})", flush=True)
            return decl_cfg, declared

    raise RuntimeError(
        f"{log_prefix} cannot build a text-only skeleton: "
        f"AutoModelForCausalLM has no model class for {wrapper} "
        f"(model_type={getattr(config, 'model_type', None)!r}), and no "
        "fallback resolved either. Tried: " + "; ".join(tried) + ". "
        "Fix by registering a ModelProfile whose "
        "stage_text_only_promote_inner_model_type()/"
        "stage_text_only_strip_keys() stage a config this transformers "
        "build can load as a text CausalLM.")


def _skeleton_config_and_class(config, *, multimodal: bool,
                               log_prefix: str = "[streaming]"):
    """Pick the `(config, model_cls)` pair `_build_streaming_context`
    instantiates the empty skeleton from.

    `model_cls is AutoModelForCausalLM` means "let the auto class
    resolve it" — the historical path, returned unchanged (same config
    object) for every config the auto class can resolve, which is every
    plain text model.
    """
    from transformers import AutoModelForCausalLM

    if multimodal:
        # Declared arch so the visual tower materializes — unchanged.
        return config, _resolve_declared_model_cls(config,
                                                   AutoModelForCausalLM)
    if _auto_causal_lm_can_resolve(config):
        return config, AutoModelForCausalLM
    # Text-only path, wrapper (e.g. vision-language) top-level config.
    return _resolve_text_only_skeleton(config, log_prefix=log_prefix)


def _find_visual_module(model) -> tuple[Any | None, str]:
    """Return (visual_module, dotted_prefix) if the model has a visual
    tower; (None, '') otherwise. Handles the v5 multimodal umbrella
    layout (`model.model.visual`) and a few common variants."""
    import torch.nn as nn
    # Most common: `model.model.visual` (Qwen3_5MoeModel.visual)
    cand = getattr(model, "model", None)
    if cand is not None:
        vis = getattr(cand, "visual", None)
        if isinstance(vis, nn.Module):
            return vis, "model.visual"
    # Fallback: top-level `model.visual` (some arch variants)
    vis = getattr(model, "visual", None)
    if isinstance(vis, nn.Module):
        return vis, "visual"
    return None, ""


def _module_has_meta_tensors(module: nn.Module) -> bool:
    return any(
        getattr(t, "is_meta", False)
        for t in (
            *module.parameters(recurse=True),
            *module.buffers(recurse=True),
        )
    )


def _build_streaming_context(model_path: str, *,
                             device: torch.device, dtype: torch.dtype,
                             offload_folder: str,
                             cache_headroom_gb: float | None = None,
                             max_cache_slots: int | None = None,
                             prefetch_workers: int | str | None = None,
                             prefetch_min_available_gb: float | str | None = None,
                             log_prefix: str = "[streaming]",
                             multimodal: bool = False,
                             visual_requires_grad: bool = False,
                             safetensors_backend: str | None = None,
                             ) -> StreamingContext:
    """One-time setup: AutoConfig + empty skeleton, then manually
    materialize only the always-resident head pieces. Decoder layers
    stay on meta until PrismaQuant streams them from safetensors.

    When `multimodal=True`:
      - Stages via `stage_multimodal` (preserves vision_config).
      - Instantiates via `config.architectures[0]` (declared arch) so the
        visual tower actually materializes — bypasses
        AutoModelForCausalLM's silent text-only downgrade.
      - After the skeleton is built, materializes the head and visual
        tower onto `device` (small — 2-3 GB even at 122B scale). Body
        still streams.
      - If `visual_requires_grad=True`, flips `.requires_grad_(True)` on
        every visual Linear's weight so Fisher backward hooks fire when
        `run_multimodal_visual_probe_pass` drives the combined forward
        (pixel_values → visual_tower → merged inputs_embeds → streamed
        body → lm_head → CE).

    When `multimodal=False` (the default) the skeleton comes from
    `AutoModelForCausalLM.from_config` exactly as before, except that a
    top-level config the auto class cannot resolve at all — a
    vision-language *wrapper* config — falls back to
    `_resolve_text_only_skeleton` instead of raising. No visual tower is
    materialized on this path either way."""
    if max_cache_slots is not None:
        if (
            isinstance(max_cache_slots, bool)
            or not isinstance(max_cache_slots, int)
            or max_cache_slots < 1
        ):
            raise ValueError("max_cache_slots must be an integer >= 1 or None")
    resolved_safetensors_backend = resolve_safetensors_backend(
        safetensors_backend
    )
    import psutil
    from transformers import AutoConfig, AutoModelForCausalLM

    from .sensitivity_probe import stage_multimodal, stage_text_only

    if not multimodal:
        # A family with no <Arch>ForCausalLM auto-route (e.g. glm5_next on
        # transformers 5.16) cannot build a text-only skeleton at all; the
        # profile declares that fact and every caller inherits the flip
        # here rather than each call site threading `multimodal=True`.
        from .model_profiles import detect_profile

        _profile = detect_profile(model_path)
        if _profile.requires_multimodal_skeleton():
            print(f"{log_prefix} profile {_profile.name} has no text-only "
                  "skeleton route; using the multimodal construction",
                  flush=True)
            multimodal = True

    bypass_hf_fp8_rewrite = False
    if multimodal:
        staged = stage_multimodal(model_path)
    else:
        bypass_hf_fp8_rewrite = _bypass_hf_fp8_module_rewrite(model_path)
        staged = stage_text_only(model_path)
        if bypass_hf_fp8_rewrite:
            print(f"{log_prefix} manual meta streaming load avoids HF fp8 "
                  "module rewrite; PrismaQuant will apply weight_scale_inv "
                  "during layer loads", flush=True)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)

    config, model_cls = _skeleton_config_and_class(
        config, multimodal=multimodal, log_prefix=log_prefix)

    with _mask_cuda_queries_during_meta_init(log_prefix):
        with init_empty_weights():
            if model_cls is AutoModelForCausalLM:
                skeleton = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True)
            else:
                skeleton = model_cls._from_config(config)
    skel_base, skel_layers = _get_layer_list(skeleton)
    base_prefix = _resolve_base_prefix(skeleton, skel_base)
    num_layers = len(skel_layers)

    # Find the visual module on the skeleton so we know which names to
    # keep resident in device_map. We rebuild these after `from_pretrained`
    # on the real model anyway — skeleton lookup only tells us the path.
    _skel_visual, skel_visual_prefix = _find_visual_module(skeleton)

    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."

    resident_device = 0 if device.type == "cuda" else "cpu"

    os.makedirs(offload_folder, exist_ok=True)
    t0 = time.time()
    print(f"{log_prefix} base_prefix={base_prefix!r}  layers={num_layers}  "
          f"head_resident_on={resident_device}  offload={offload_folder}  "
          f"multimodal={multimodal}  visual_prefix={skel_visual_prefix or 'n/a'}  "
          f"safetensors_backend={resolved_safetensors_backend}",
          flush=True)

    model = skeleton
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)

    weight_shard, weight_ckpt = _build_weight_map(model_path, multimodal=multimodal)
    # Native-FP8 source dequant map. Populated only for checkpoints that
    # ship `.weight_scale_inv` siblings (MiniMax-M2/M2.7, DeepSeek-V3).
    # Empty dict for plain BF16 checkpoints — `_read_layer_to_device`
    # then skips the dequant pass entirely. This map is THE fix for the
    # probe/cost/export mismatch where the streaming loader previously
    # cast fp8 codes to bf16 without applying the 128x128 block scale,
    # leaving every downstream pass operating on raw codes (range ±448)
    # instead of true weights (range ±0.2).
    fp8_scale_inv_map = _build_fp8_scale_inv_map(
        model_path, multimodal=multimodal)
    if fp8_scale_inv_map:
        print(f"{log_prefix} fp8 scale_inv map: {len(fp8_scale_inv_map)} "
              f"weights will be dequanted inline at layer-load",
              flush=True)

    head_pfxs = _head_prefixes(model, base_prefix)
    loaded_head = _materialize(
        model,
        head_pfxs,
        weight_shard,
        weight_ckpt,
        device,
        dtype,
        fp8_scale_inv_map=fp8_scale_inv_map,
    )
    # Weight tying: a `tie_word_embeddings` checkpoint ships no
    # `lm_head.weight`, so `_materialize` above has nothing to install and
    # the head stays on meta — the first `model.lm_head(...)` (probe
    # Phase-2 CE) or `m.weight.to(device)` (cost stage) then dies with
    # "Cannot copy out of meta tensor". Resolve the alias through
    # transformers' own embedding accessors (no name hardcoded: the VL
    # wrapper's `model.language_model.embed_tokens` resolves like the
    # plain `model.embed_tokens`).
    resolve_tied_output_embedding(model, log_prefix=log_prefix)
    _init_rotary_inplace(base_model, device, dtype)
    print(f"{log_prefix} head materialized ({loaded_head} tensors, "
          f"rotary re-init) in {time.time()-t0:.1f}s", flush=True)

    # Constructor-derived NON-PERSISTENT head buffers (e.g. gemma4_unified's
    # `embed_scale = sqrt(hidden)`) are absent from the checkpoint, so
    # `_materialize` never assigns them and they stay on `meta` — and
    # PrismaQuant globally no-ops `_initialize_weights` (prismaquant/__init__),
    # so the modeling's `_init_weights` that would set them never runs. The
    # first forward op (`embed_tokens(ids)` multiplies by `embed_scale`) then
    # faults "Tensor on device meta". Re-create such buffers on `device` from
    # the owning module's retained python scalar (`scalar_<attr>`). Generic
    # (any arch following this pattern); scoped to non-`layers` modules since
    # streaming decoder buffers load per shard. Persistent buffers (e.g.
    # `layer_scalar`) come from the checkpoint and are untouched.
    _meta_fixed = 0
    for _bname, _buf in list(base_model.named_buffers(recurse=True)):
        if _buf is None or not _buf.is_meta or _bname.split(".", 1)[0] == "layers":
            continue
        _mod_name, _, _attr = _bname.rpartition(".")
        _owner = base_model.get_submodule(_mod_name) if _mod_name else base_model
        if _attr not in getattr(_owner, "_non_persistent_buffers_set", set()):
            continue  # persistent buffers are loaded from the checkpoint
        _scalar = getattr(_owner, "scalar_" + _attr, None)
        if _scalar is None:
            continue
        _owner.register_buffer(
            _attr, torch.tensor(_scalar, device=device, dtype=_buf.dtype),
            persistent=False)
        _meta_fixed += 1
    _stuck = [n for n, b in base_model.named_buffers(recurse=True)
              if b is not None and b.is_meta and n.split(".", 1)[0] != "layers"]
    if _stuck:
        raise RuntimeError(
            f"{log_prefix} head buffers left on meta after materialization "
            f"(no scalar_ init source): {_stuck[:8]} — extend the "
            f"non-persistent-buffer sweep in _build_streaming_context")
    if _meta_fixed:
        print(f"{log_prefix} materialized {_meta_fixed} non-persistent head "
              f"buffer(s) off meta (e.g. embed_scale)", flush=True)

    # Locate the visual module on the meta skeleton. When multimodal is
    # set, fully materialize the visual tower onto `device`; body
    # layers remain meta and stream per shard.
    visual_module = None
    visual_prefix: str | None = None
    if multimodal:
        visual_module, visual_prefix = _find_visual_module(model)
        if visual_module is not None and visual_prefix:
            remove_hook_from_module(visual_module, recurse=True)
            vis_keys = [k for k in weight_shard if k.startswith(visual_prefix + ".")]
            # Load all visual tensors from safetensors onto device.
            tensors = _read_layer_to_device(
                visual_prefix + ".",
                weight_shard, weight_ckpt, dtype, device,
                fp8_scale_inv_map=fp8_scale_inv_map,
                safetensors_backend=resolved_safetensors_backend)
            print(f"{log_prefix} materializing visual tower: "
                  f"{len(tensors)}/{len(vis_keys)} tensors -> {device}", flush=True)
            if _module_has_meta_tensors(visual_module):
                visual_module.to_empty(device=device, recurse=True)
            for model_name, t in tensors.items():
                install_dtype = t.dtype if t.is_floating_point() else None
                set_module_tensor_to_device(
                    model, model_name, device, value=t, dtype=install_dtype)
            # Some visual towers carry non-checkpoint buffers initialized by
            # the module constructor. Keep them colocated with checkpoint
            # tensors before the multimodal streaming probe calls visual
            # helpers such as get_image_features.
            visual_module.to(device=device, dtype=dtype)
            if visual_requires_grad:
                # Enable grad on every Linear's weight + bias so backward
                # hooks fire on the reverse sweep. Embeddings and norms
                # stay frozen (no Fisher tracked for those).
                import torch.nn as nn
                n_grad = 0
                for n, m in visual_module.named_modules():
                    if isinstance(m, nn.Linear):
                        for p in m.parameters(recurse=False):
                            p.requires_grad_(True)
                            n_grad += 1
                print(f"{log_prefix} visual: enabled grad on "
                      f"{n_grad} Linear params", flush=True)
    print(f"{log_prefix} model ready in {time.time()-t0:.1f}s", flush=True)

    print(f"{log_prefix} building install resolvers for {num_layers} layers ...",
          flush=True)
    t_res = time.time()
    install_resolvers = [
        _build_install_resolver(model, f"{layers_prefix}{L}".rstrip("."))
        for L in range(num_layers)
    ]
    print(f"{log_prefix} resolvers built: "
          f"{sum(len(r) for r in install_resolvers)} tensors across "
          f"{num_layers} layers in {time.time()-t_res:.1f}s", flush=True)

    free_bytes = psutil.virtual_memory().available
    # Resolve headroom: env override > explicit arg > autoscale > legacy 75 GB default.
    resolved_headroom_gb = cache_headroom_gb
    autoscale_diag = None
    if resolved_headroom_gb is None:
        env_val = os.environ.get("CACHE_HEADROOM_GB")
        if env_val not in (None, "", "auto", "AUTO"):
            resolved_headroom_gb = float(env_val)
        else:
            try:
                from .autoscale import pick_cache_headroom_gb
                resolved_headroom_gb, autoscale_diag = pick_cache_headroom_gb(
                    model_path,
                    layers_per_shard=int(os.environ.get("LAYERS_PER_SHARD", "1") or 1)
                        if str(os.environ.get("LAYERS_PER_SHARD", "")).isdigit() else 1,
                    nsamples=int(os.environ.get("NSAMPLES", "32")),
                    seqlen=int(os.environ.get("SEQLEN", "1024")),
                )
            except Exception as e:
                print(f"{log_prefix} autoscale failed ({e!r}); falling back to 75 GB headroom",
                      flush=True)
                resolved_headroom_gb = 75.0
    cache_bytes = max(int(free_bytes) - int(resolved_headroom_gb * 1024 ** 3),
                      8 * 1024 ** 3)
    layer_cache = LayerCache(
        max_bytes=cache_bytes,
        max_entries=max_cache_slots,
    )
    # v20 step 3+4: enable dynamic budget with the same headroom reserve
    # used to size the static max. The cache shrinks when host memory
    # tightens (other processes growing, gradient transients) and grows
    # back to static_max when slack returns.
    layer_cache.configure_dynamic_budget(int(resolved_headroom_gb * 1024 ** 3))
    src = "explicit" if autoscale_diag is None else "autoscaled"
    print(f"{log_prefix} layer cache budget={cache_bytes/(1024**3):.1f} GB "
          f"(free={free_bytes/(1024**3):.1f} GB, headroom={resolved_headroom_gb:.1f} GB, "
          f"dynamic_reserve={resolved_headroom_gb:.1f} GB, {src})",
          flush=True)
    if autoscale_diag is not None:
        print(f"{log_prefix}   autoscale: shard_working={autoscale_diag['shard_working_gb']:.1f} GB "
              f"+ safety={autoscale_diag['safety_gb']:.1f} GB "
              f"(lps={autoscale_diag['layers_per_shard']})", flush=True)

    estimated_layer_bytes, layer_bytes = _estimate_layer_cache_bytes(
        weight_shard=weight_shard,
        weight_ckpt=weight_ckpt,
        layers_prefix=layers_prefix,
        num_layers=num_layers,
        target_dtype=dtype,
        fp4_experts=declared_fp4_expert_dtype(model_path),
        safetensors_backend=resolved_safetensors_backend,
    )
    worker_count, worker_src = _auto_prefetch_workers(
        cache_bytes, estimated_layer_bytes, requested=prefetch_workers)
    if max_cache_slots is not None:
        worker_count = min(worker_count, max_cache_slots)
        worker_src = f"{worker_src}, capped by max_cache_slots"
    min_available_bytes, min_available_src = _auto_prefetch_min_available_bytes(
        estimated_layer_bytes, requested=prefetch_min_available_gb)
    cache_slots = (
        int(cache_bytes // estimated_layer_bytes)
        if estimated_layer_bytes > 0 else 0
    )
    memory_slots = 0
    if estimated_layer_bytes > 0:
        memory_slots = max(
            0, int((free_bytes - min_available_bytes) // estimated_layer_bytes))
    print(f"{log_prefix} prefetch auto: workers={worker_count} "
          f"({worker_src}), cache_slots={cache_slots}, "
          f"max_cache_slots={max_cache_slots}, "
          f"memory_slots={memory_slots}, "
          f"est_layer={estimated_layer_bytes/(1024**3):.1f} GB, "
          f"min_avail={min_available_bytes/(1024**3):.1f} GB "
          f"({min_available_src})", flush=True)

    prefetch_pool = ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="prefetch")

    return StreamingContext(
        model=model, base_model=base_model, layers=layers,
        layers_prefix=layers_prefix, num_layers=num_layers,
        install_resolvers=install_resolvers,
        weight_shard=weight_shard, weight_ckpt=weight_ckpt,
        layer_cache=layer_cache, prefetch_pool=prefetch_pool,
        device=device, dtype=dtype, offload_folder=offload_folder,
        visual_module=visual_module,
        visual_prefix=visual_prefix,
        multimodal=multimodal,
        fp8_scale_inv_map=fp8_scale_inv_map,
        estimated_layer_bytes=estimated_layer_bytes,
        prefetch_workers=worker_count,
        prefetch_min_available_bytes=min_available_bytes,
        safetensors_backend=resolved_safetensors_backend,
        expert_packer=_build_expert_packer(model, weight_ckpt),
        concat_merger=_build_concat_merger(model, weight_ckpt),
    )
