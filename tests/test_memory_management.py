import importlib
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant import iterate_perturbed_allocation as ipa
from prismaquant import memory_management as mm
from prismaquant import propagated_cost as pc
from prismaquant.perturbed_x_cache import PerturbedActivationCache
from prismaquant.propagated_cost import CUDAGraphRegistry, _CUDAGraphEntry
from prismaquant.memory_management import enforce_gpu_memory_budget, report_graph_memory


_NOCLONE_OVERRIDE_FRAGMENT = (
    "PRISMAQUANT_GRAPH_OUTPUT_CLONE=0 is unsafe with "
    "PRISMAQUANT_GRAPH_SHARED_POOL=1"
)


class _ManyLinear(nn.Module):
    def __init__(self, count: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(64, 64, bias=False) for _ in range(count)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_frozen_weight_cache_lru_eviction(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_FROZEN_WEIGHT_CACHE_MAX_ENTRIES", "3")
    monkeypatch.setenv("PRISMAQUANT_MAX_GPU_MEM_GB", "0")
    model = _ManyLinear(8).eval()
    assignment = {f"layers.{idx}": "BF16" for idx in range(8)}
    builder = PerturbedActivationCache(
        model,
        assignment,
        tmp_path,
        input_rows=0,
        cal_hash="fixed",
    )

    with builder.frozen_weight_cache():
        pass

    logical_keys = [
        (name, fmt)
        for name, fmt, *_ in builder._frozen_weight_format_cache.keys()
    ]
    assert logical_keys == [(f"layers.{idx}", "BF16") for idx in range(5, 8)]
    assert builder._frozen_weight_cache_evictions == 5


def test_cuda_graph_cache_lru_eviction(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH", "3")
    registry = CUDAGraphRegistry(label="test-graph-lru", max_entries=10)
    for idx in range(8):
        registry.entries[(idx,)] = _CUDAGraphEntry(
            graph=SimpleNamespace(replay=lambda: None),
            static_args=(),
            static_kwargs={},
            static_output=None,
        )

    registry._evict_if_needed()

    assert list(registry.entries.keys()) == [(5,), (6,), (7,)]
    assert registry.eviction_count == 5


def test_cuda_graph_capture_failure_cleanup_resets_graph(monkeypatch):
    calls = []
    registry = CUDAGraphRegistry(label="test-graph-cleanup", max_entries=10)
    graph = SimpleNamespace(reset=lambda: calls.append("reset"))
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: calls.append(("sync", str(device))),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    registry._cleanup_failed_capture(
        graph,
        torch.device("cuda:0"),
        "test-capture",
    )

    assert calls == ["reset", ("sync", "cuda:0"), "empty"]


def test_cuda_graph_registries_share_lazy_pool(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setenv("PRISMAQUANT_GRAPH_SHARED_POOL", "1")
    monkeypatch.setattr(pc, "_PRISMAQUANT_GRAPH_POOL", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "graph_pool_handle",
        lambda: calls.append("pool") or sentinel,
        raising=False,
    )

    first = CUDAGraphRegistry(label="pool-a")
    second = CUDAGraphRegistry(label="pool-b")

    assert first.graph_pool() is sentinel
    assert second.graph_pool() is sentinel
    assert pc.get_prismaquant_graph_pool() is sentinel
    assert calls == ["pool"]
    assert first.graph_pool_id() == second.graph_pool_id()


def test_cuda_graph_shared_pool_env_can_disable(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_GRAPH_SHARED_POOL", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "graph_pool_handle",
        lambda: pytest.fail("disabled shared pool should not initialize"),
        raising=False,
    )

    registry = CUDAGraphRegistry(label="private-pool")

    assert registry.graph_pool() is None
    assert registry.graph_pool_id() == "private"


def test_output_clone_overridden_when_shared_pool_enabled(monkeypatch, capsys):
    monkeypatch.setenv("PRISMAQUANT_GRAPH_SHARED_POOL", "1")
    monkeypatch.setenv("PRISMAQUANT_GRAPH_OUTPUT_CLONE", "0")
    monkeypatch.setattr(pc, "_NOCLONE_OVERRIDE_WARNED", False)
    value = torch.tensor([1.0, 2.0])

    cloned = pc._clone_cuda_graph_output(value)

    captured = capsys.readouterr()
    assert cloned is not value
    assert torch.equal(cloned, value)
    assert _NOCLONE_OVERRIDE_FRAGMENT in captured.err


def test_output_clone_override_warning_emitted_once(monkeypatch, capsys):
    monkeypatch.setenv("PRISMAQUANT_GRAPH_SHARED_POOL", "1")
    monkeypatch.setenv("PRISMAQUANT_GRAPH_OUTPUT_CLONE", "0")
    monkeypatch.setattr(pc, "_NOCLONE_OVERRIDE_WARNED", False)
    value = torch.tensor([1.0, 2.0])

    for _ in range(5):
        cloned = pc._clone_cuda_graph_output(value)
        assert cloned is not value

    captured = capsys.readouterr()
    assert captured.err.count(_NOCLONE_OVERRIDE_FRAGMENT) == 1


def test_output_clone_skipped_when_only_clone_disabled(monkeypatch, capsys):
    monkeypatch.setenv("PRISMAQUANT_GRAPH_SHARED_POOL", "0")
    monkeypatch.setenv("PRISMAQUANT_GRAPH_OUTPUT_CLONE", "0")
    monkeypatch.setattr(pc, "_NOCLONE_OVERRIDE_WARNED", False)
    value = torch.tensor([1.0, 2.0])

    output = pc._clone_cuda_graph_output(value)

    captured = capsys.readouterr()
    assert output is value
    assert _NOCLONE_OVERRIDE_FRAGMENT not in captured.err


def test_cuda_graph_output_clone_can_return_static_replay_tensor(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_GRAPH_SHARED_POOL", "0")
    monkeypatch.setenv("PRISMAQUANT_GRAPH_OUTPUT_CLONE", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    registry = CUDAGraphRegistry(label="output-alias", max_entries=4)
    arg = torch.ones(2)
    static_arg = torch.empty_like(arg)
    static_output = torch.zeros(2)
    full_key = (
        registry.label,
        "alias",
        ("key",),
        pc._tensor_tree_signature((arg,)),
        pc._tensor_tree_signature({}),
    )

    registry.entries[full_key] = _CUDAGraphEntry(
        graph=SimpleNamespace(replay=lambda: static_output.add_(1.0)),
        static_args=(static_arg,),
        static_kwargs={},
        static_output=static_output,
    )

    first = registry.run(
        "alias",
        ("key",),
        lambda x: x + 1,
        arg,
        enabled=True,
        device=torch.device("cuda"),
    )
    second = registry.run(
        "alias",
        ("key",),
        lambda x: x + 1,
        arg,
        enabled=True,
        device=torch.device("cuda"),
    )

    assert first is static_output
    assert second is static_output
    assert first is second
    assert static_output.tolist() == [2.0, 2.0]


def test_cuda_memory_info_uses_host_available_for_integrated_uma(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_UMA_MEMORY_INFO", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (2, 100))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(is_integrated=1),
    )
    monkeypatch.setattr(mm, "_host_memory_info", lambda: (60, 128))

    assert mm.cuda_memory_info(torch.device("cuda:0")) == (60, 100)


def test_cuda_memory_info_can_disable_uma_host_available(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_UMA_MEMORY_INFO", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (2, 100))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(is_integrated=1),
    )
    monkeypatch.setattr(mm, "_host_memory_info", lambda: (60, 128))

    assert mm.cuda_memory_info(torch.device("cuda:0")) == (2, 100)


def test_graph_memory_audit_reports_registered_graphs(monkeypatch, capsys):
    monkeypatch.setenv("PRISMAQUANT_GRAPH_AUDIT", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    registry = CUDAGraphRegistry(label="audit-registry", max_entries=4)
    registry.entries[("entry",)] = _CUDAGraphEntry(
        graph=SimpleNamespace(replay=lambda: None),
        static_args=(torch.ones(2, dtype=torch.float32),),
        static_kwargs={},
        static_output=torch.ones(3, dtype=torch.float32),
    )

    report_graph_memory("unit")

    captured = capsys.readouterr()
    assert "label=unit registry=audit-registry" in captured.err
    assert "entries=1" in captured.err
    assert "static_bytes=20" in captured.err


def test_phase_boundary_cleanup_called(monkeypatch):
    calls = {"empty_cache": 0}

    def _empty_cache():
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", _empty_cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    ipa._phase_boundary_cleanup("l2_to_l3")

    assert calls["empty_cache"] == 1


def test_memory_budget_evicts_when_low(monkeypatch):
    gib = 1024 ** 3
    mem_info = iter([(0, 2 * gib), (1536 * 1024 ** 2, 2 * gib)])

    class _Evictor:
        def __init__(self):
            self.entries = OrderedDict((idx, idx) for idx in range(3))
            self.evicted = []

        def evict_oldest_for_memory_budget(self):
            if not self.entries:
                return False
            key, _value = self.entries.popitem(last=False)
            self.evicted.append(key)
            return True

    evictor = _Evictor()
    monkeypatch.setenv("PRISMAQUANT_MAX_GPU_MEM_GB", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: next(mem_info))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    evicted = enforce_gpu_memory_budget([evictor], reason="test")

    assert evicted == 1
    assert evictor.evicted == [0]


def test_triton_warmup_compiles_kernel(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_NVFP4_FUSED_JIT_WARMUP", raising=False)
    module = importlib.import_module("prismaquant.kernels.nvfp4_fused")
    state = module.nvfp4_fused_warmup_state()

    assert state["attempted"] is True
    if not torch.cuda.is_available():
        assert state["skipped_reason"] == "cuda_unavailable"
        pytest.skip("CUDA unavailable")
    assert state["compiled"] is True
    assert (8, 8, 64, 16, 32, 64) in state["compiled_signatures"]
