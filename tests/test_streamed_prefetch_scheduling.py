"""Prefetch scheduling for the sequential streamed layer walk.

Regression cover for the GLM-5.3-Flash loader starvation (2026-08-26): the
probe's phase-1/phase-3 sweeps read 891 GB for a walk that needed ~520 GB,
because a prefetch whose `put()` the dynamic budget refused — or whose
`pinned_until_read` entry a pressure trim evicted — was silently discarded
and re-read synchronously by the consumer.

No GPU and no model: these drive `StreamingContext`'s scheduling logic over
a stub layer reader.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import prismaquant.streaming_model as streaming_model
from prismaquant.layer_streaming import LayerCache
from prismaquant.streaming_model import StreamingContext

LAYER_BYTES = 1 << 20  # 1 MiB of "layer"


def _layer_tensors(L: int) -> dict[str, torch.Tensor]:
    return {
        f"model.layers.{L}.weight": torch.full(
            (LAYER_BYTES // 4,), float(L), dtype=torch.float32,
        )
    }


def _make_ctx(
    monkeypatch,
    *,
    num_layers: int = 8,
    cache_bytes: int = 64 * LAYER_BYTES,
    max_entries: int | None = None,
    workers: int = 4,
    pressure_floor: int = 0,
    reads: list[int] | None = None,
) -> StreamingContext:
    """A StreamingContext whose layer reader is a counted stub."""
    read_log = reads if reads is not None else []

    def fake_read(prefix, *args, **kwargs):
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        read_log.append(L)
        return _layer_tensors(L)

    monkeypatch.setattr(streaming_model, "_read_layer_to_device", fake_read)

    ctx = object.__new__(StreamingContext)
    ctx.layers_prefix = "model.layers."
    ctx.num_layers = num_layers
    ctx.weight_shard = {}
    ctx.weight_ckpt = {}
    ctx.device = torch.device("cpu")
    ctx.dtype = torch.float32
    ctx.fp8_scale_inv_map = {}
    ctx.expert_packer = None
    ctx.concat_merger = None
    ctx.estimated_layer_bytes = LAYER_BYTES
    ctx.prefetch_workers = workers
    ctx.prefetch_min_available_bytes = pressure_floor
    ctx.prefetch_memory_skips = 0
    ctx.prefetch_delivered_unretained = 0
    ctx.prefetch_released_stale = 0
    ctx._inflight = {}
    ctx._inflight_lock = threading.Lock()
    ctx._last_installed = None
    ctx._walk_step = 0
    ctx.layer_cache = LayerCache(max_bytes=cache_bytes, max_entries=max_entries)
    ctx.max_cache_slots = ctx.layer_cache.max_entries
    ctx.prefetch_pool = ThreadPoolExecutor(max_workers=workers)
    ctx._read_log = read_log
    return ctx


def _drain(ctx: StreamingContext) -> None:
    ctx.prefetch_pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# 1. The bug itself: a refused put must not cost a second read.
def test_refused_put_is_still_delivered_and_never_re_read(monkeypatch):
    reads: list[int] = []
    ctx = _make_ctx(monkeypatch, reads=reads)
    # Budget that admits nothing: exactly the GLM condition, where
    # `_effective_max()` fell below one layer while the walk still needed it.
    monkeypatch.setattr(ctx.layer_cache, "_effective_max", lambda: 0)

    fut = ctx.schedule_prefetch(3)
    assert fut is not None
    fut.result()
    assert not ctx.layer_cache.peek(3), "budget was supposed to refuse the put"

    tensors, src = ctx.ensure_loaded(3)

    assert src == "wait", "a completed prefetch must be delivered, not re-read"
    assert torch.equal(tensors["model.layers.3.weight"][0], torch.tensor(3.0))
    assert reads == [3], f"layer 3 was read {len(reads)} times: {reads}"
    assert ctx.prefetch_delivered_unretained == 1
    _drain(ctx)


def test_pre_fix_lever_reproduces_the_double_read(monkeypatch):
    """`PRISMAQUANT_PREFETCH_DELIVERY=0` must restore the old behaviour, so
    the A/B that measured the fix stays reproducible."""
    monkeypatch.setenv("PRISMAQUANT_PREFETCH_DELIVERY", "0")
    reads: list[int] = []
    ctx = _make_ctx(monkeypatch, reads=reads)
    monkeypatch.setattr(ctx.layer_cache, "_effective_max", lambda: 0)

    ctx.schedule_prefetch(3).result()
    _tensors, src = ctx.ensure_loaded(3)

    assert src == "cold"
    assert reads == [3, 3]
    _drain(ctx)


# ---------------------------------------------------------------------------
# 2. The enqueue itself: a sequential walk keeps the window in flight.
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_sequential_walk_keeps_the_lookahead_window_enqueued(
    monkeypatch, direction,
):
    reads: list[int] = []
    ctx = _make_ctx(monkeypatch, num_layers=8, reads=reads)
    order = range(8) if direction == "forward" else range(7, -1, -1)

    seen_inflight = []
    srcs = []
    for L in order:
        tensors, src = ctx.ensure_loaded(L)
        srcs.append(src)
        # `install()` normally drives this; there is no live model here.
        ctx._top_up_prefetch(L)
        with ctx._inflight_lock:
            seen_inflight.append(sorted(ctx._inflight))
        del tensors

    # Layer 0 of the walk cannot be prefetched (nothing has run yet), and the
    # second layer only gets a window once the stride is known. Everything
    # after that must be served without touching the cold loader.
    assert srcs[0] == "cold"
    assert "cold" not in srcs[2:], f"cold reads mid-walk: {srcs}"
    assert any(seen_inflight[2:]), "lookahead window never got enqueued"
    assert len(reads) == len(set(reads)) == 8, f"re-read layers: {reads}"
    _drain(ctx)


def test_random_access_walk_does_not_speculate(monkeypatch):
    """Polish flips and isolated re-runs jump around; extrapolating a
    direction from them would prefetch layers nobody asked for."""
    reads: list[int] = []
    ctx = _make_ctx(monkeypatch, num_layers=16, reads=reads)
    for L in (0, 7, 2, 11):
        ctx.ensure_loaded(L)
        ctx._top_up_prefetch(L)
    _drain(ctx)
    assert sorted(set(reads)) == [0, 2, 7, 11]


# ---------------------------------------------------------------------------
# 3. Admission: unfinished reads are bounded by real memory, not by the static
#    cache budget; completed delivery aliases are already charged there.
def test_admission_bound_limits_unclaimed_prefetches(monkeypatch):
    ctx = _make_ctx(monkeypatch, num_layers=32)
    # Room for two layer-sized speculative reads.
    monkeypatch.setattr(ctx, "affordable_prefetch_slots", lambda: 2)
    block = threading.Event()

    def blocking_read(prefix, *args, **kwargs):
        block.wait(5)
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        return _layer_tensors(L)

    monkeypatch.setattr(streaming_model, "_read_layer_to_device", blocking_read)
    for L in range(8):
        ctx.schedule_prefetch(L)
    with ctx._inflight_lock:
        assert len(ctx._inflight) == 2
    assert ctx.prefetch_memory_skips == 6
    block.set()
    _drain(ctx)


def test_admission_reserves_pending_reads_not_completed_cache_aliases(monkeypatch):
    """Completed opening futures remain claimable aliases, not new reads.

    Their tensors are already charged to MemAvailable and retained in the
    LayerCache. They must not consume the additional-read budget, while queued
    or running futures still consume exactly one reservation each.
    """
    ctx = _make_ctx(monkeypatch, num_layers=8)
    monkeypatch.setattr(ctx, "affordable_prefetch_slots", lambda: 2)

    for L in (0, 1):
        fut = ctx.schedule_prefetch(L)
        assert fut is not None
        fut.result()
        assert ctx.layer_cache.peek(L)
    with ctx._inflight_lock:
        assert set(ctx._inflight) == {0, 1}
        assert all(fut.done() for fut in ctx._inflight.values())

    block = threading.Event()

    def blocking_read(prefix, *args, **kwargs):
        block.wait(5)
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        return _layer_tensors(L)

    monkeypatch.setattr(streaming_model, "_read_layer_to_device", blocking_read)
    assert ctx.schedule_prefetch(2) is not None
    assert ctx.schedule_prefetch(3) is not None
    assert ctx.schedule_prefetch(4) is None

    with ctx._inflight_lock:
        assert set(ctx._inflight) == {0, 1, 2, 3}
        assert sum(not fut.done() for fut in ctx._inflight.values()) == 2
    assert ctx.prefetch_memory_skips == 1
    block.set()
    _drain(ctx)


def test_lookahead_is_clamped_by_affordable_memory(monkeypatch):
    ctx = _make_ctx(monkeypatch, cache_bytes=64 * LAYER_BYTES)
    monkeypatch.setattr(ctx, "affordable_prefetch_slots", lambda: 0)
    # Static budget alone would say 63; memory says one layer at a time.
    assert ctx.suggest_prefetch_lookahead() == 1
    monkeypatch.setattr(ctx, "affordable_prefetch_slots", lambda: 4)
    assert ctx.suggest_prefetch_lookahead() == 4
    _drain(ctx)


def test_turnaround_releases_stale_prefetch_bytes(monkeypatch):
    """Phase-1 forward -> phase-3 reverse: futures now hold layer bytes until
    claimed, so the turnaround has to hand back what it queued ahead."""
    ctx = _make_ctx(monkeypatch, num_layers=8)
    for L in (5, 6, 7):
        ctx.schedule_prefetch(L)
    for fut in list(ctx._inflight.values()):
        fut.result()
    assert len(ctx._inflight) == 3

    # Walk arrives at 4 heading down.
    ctx._last_installed = 5
    ctx._walk_step = -1
    ctx._top_up_prefetch(4)

    with ctx._inflight_lock:
        assert 6 not in ctx._inflight and 7 not in ctx._inflight
    assert ctx.prefetch_released_stale >= 2
    _drain(ctx)


def test_repeated_endpoint_preserves_seeded_reverse_prefetch(monkeypatch):
    """Forward N-1 -> reverse N-1 must not discard the reverse seed.

    The probe installs the top layer at the end of phase 1, prefetches N-2,
    then installs the same top layer to begin phase 3.  A zero delta is a
    phase boundary, not evidence that the old +1 direction should continue.
    """
    reads: list[int] = []
    ctx = _make_ctx(monkeypatch, num_layers=8, reads=reads)
    # Keep the completed future as the only delivery path.  If top-up drops
    # it, ensure_loaded must perform a second source read and the test catches
    # the exact NFS amplification seen by the production schedule.
    monkeypatch.setattr(ctx.layer_cache, "_effective_max", lambda: 0)
    ctx._last_installed = 7
    ctx._walk_step = 1

    reverse_seed = ctx.schedule_prefetch(6)
    assert reverse_seed is not None
    reverse_seed.result()
    assert not ctx.layer_cache.peek(6)

    ctx._top_up_prefetch(7)
    _tensors, src = ctx.ensure_loaded(6)

    assert src == "wait"
    assert reads == [6]
    _drain(ctx)


# ---------------------------------------------------------------------------
# 4. Pressure trim must not evict a layer the walk pinned for imminent use.
def test_pressure_trim_prefers_unpinned_victims(monkeypatch):
    cache = LayerCache(max_bytes=64 * LAYER_BYTES)
    cache.put(0, _layer_tensors(0))                     # cold, unpinned
    cache.put(1, _layer_tensors(1), pinned_until_read=True)

    # One layer short of the floor on entry; the trim's post-eviction
    # re-check then sees the pressure relieved, as it would on a real box.
    calls = {"n": 0}

    def _vm():
        calls["n"] += 1
        avail = 0 if calls["n"] == 1 else 2 * LAYER_BYTES
        return type("_VM", (), {"available": avail})()

    monkeypatch.setattr("psutil.virtual_memory", _vm)
    cache._pressure_threshold_bytes = LAYER_BYTES  # need one layer freed
    cache._maybe_pressure_shrink()

    assert not cache.peek(0), "unpinned LRU entry should have been the victim"
    assert cache.peek(1), "prefetched-but-unread layer was evicted first"
    assert cache.evicted_pinned == 0


def test_pressure_trim_clears_the_pin_when_it_does_evict(monkeypatch):
    cache = LayerCache(max_bytes=64 * LAYER_BYTES)
    cache.put(0, _layer_tensors(0), pinned_until_read=True)

    class _VM:
        available = 0

    monkeypatch.setattr("psutil.virtual_memory", lambda: _VM())
    cache._pressure_threshold_bytes = LAYER_BYTES
    cache._maybe_pressure_shrink()

    assert not cache.peek(0)
    assert 0 not in cache._pinned_until_read, "stale pin leaked"
    assert cache.evicted_pinned == 1
