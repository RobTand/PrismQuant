from __future__ import annotations

import ast
from concurrent.futures import Future
import inspect
import threading
from types import SimpleNamespace

import pytest
import torch

from prismaquant.cost_streaming import build_streamed_causal_lm
from prismaquant.dsv4_aura_cb_reprice import (
    DSV4_STREAMING_CACHE_MAX_SLOTS,
    _measure_streamed,
)
from prismaquant.layer_streaming import LayerCache
from prismaquant.streaming_model import (
    StreamingContext,
    _build_streaming_context,
)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
def test_layer_cache_rejects_invalid_entry_caps(value):
    with pytest.raises(ValueError, match="max_entries"):
        LayerCache(max_bytes=1 << 30, max_entries=value)


def test_layer_cache_entry_cap_evicts_even_when_byte_budget_is_large():
    cache = LayerCache(max_bytes=1 << 30, max_entries=2)
    tensors = {
        index: {"weight": torch.zeros(4, dtype=torch.float32)}
        for index in range(3)
    }

    cache.put(0, tensors[0])
    cache.put(1, tensors[1])
    assert cache.get(0) is tensors[0]  # layer 1 becomes LRU
    cache.put(2, tensors[2])

    assert len(cache._cache) == 2
    assert cache.peek(0)
    assert not cache.peek(1)
    assert cache.peek(2)


def test_one_slot_cache_pre_evicts_before_incoming_layer_read():
    cache = LayerCache(max_bytes=1 << 30, max_entries=1)
    tensor = {"weight": torch.zeros(8, dtype=torch.float32)}
    cache.put(0, tensor, pinned_until_read=True)

    freed = cache.prepare_for_load(cache._sizeof(tensor))

    assert freed == cache._sizeof(tensor)
    assert not cache._cache
    assert not cache._pinned_until_read
    assert cache.total_bytes == 0


def test_shared_context_rejects_invalid_cap_before_model_construction():
    with pytest.raises(ValueError, match="max_cache_slots"):
        _build_streaming_context(
            "/model-that-must-not-be-opened",
            device=torch.device("cpu"),
            dtype=torch.float32,
            offload_folder="/offload",
            max_cache_slots=0,
        )


def test_one_slot_shared_context_refuses_speculative_prefetch():
    context = object.__new__(StreamingContext)
    context.max_cache_slots = 1

    assert context.schedule_prefetch(0) is None


@pytest.mark.parametrize("has_inflight", [False, True])
def test_required_prefetch_refuses_cold_loader_when_residency_is_missing(
    monkeypatch, has_inflight,
):
    import prismaquant.streaming_model as streaming_model

    context = object.__new__(StreamingContext)
    context.layer_cache = LayerCache(max_bytes=1 << 30, max_entries=2)
    context._inflight = {}
    context._inflight_lock = threading.Lock()
    if has_inflight:
        completed_without_residency = Future()
        completed_without_residency.set_result(None)
        context._inflight[0] = completed_without_residency

    cold_reads = []

    def forbidden_cold_read(*args, **kwargs):
        cold_reads.append((args, kwargs))
        raise AssertionError("required-prefetch path invoked cold loader")

    monkeypatch.setattr(
        streaming_model, "_read_layer_to_device", forbidden_cold_read,
    )

    with pytest.raises(RuntimeError, match="refusing synchronous cold"):
        context.ensure_loaded(0, require_prefetched=True)

    assert cold_reads == []


def test_cost_streaming_builder_threads_slot_cap_and_disables_lookahead(
    monkeypatch,
):
    import prismaquant.streaming_model as streaming_model

    captured = {}
    context = SimpleNamespace(
        model=object(),
        base_model=object(),
        layers=[],
        layers_prefix="model.layers.",
        num_layers=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cache_slots=1,
    )

    def fake_build_context(_model_path, **kwargs):
        captured.update(kwargs)
        return context

    monkeypatch.setattr(
        streaming_model, "_build_streaming_context", fake_build_context,
    )
    runner = build_streamed_causal_lm(
        "/model",
        device=torch.device("cpu"),
        dtype=torch.float32,
        offload_folder="/offload",
        profile=object(),
        max_cache_slots=1,
        prefetch_lookahead=8,
    )

    assert captured["max_cache_slots"] == 1
    assert runner.context is context
    assert runner.prefetch_lookahead == 0
    assert runner.require_prefetched_residency is False


def test_dsv4_measurement_hard_wires_one_source_cache_slot():
    assert DSV4_STREAMING_CACHE_MAX_SLOTS == 1
    tree = ast.parse(inspect.getsource(_measure_streamed))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_streamed_causal_lm"
    ]
    assert len(calls) == 1
    slot_cap = next(
        keyword.value
        for keyword in calls[0].keywords
        if keyword.arg == "max_cache_slots"
    )
    assert isinstance(slot_cap, ast.Name)
    assert slot_cap.id == "DSV4_STREAMING_CACHE_MAX_SLOTS"
