"""Checkpoint buffers retain their declared precision across source loads."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from prismaquant.layer_streaming import (
    LayerCache, _build_install_resolver, _materialize,
)
from prismaquant.streaming_model import StreamingContext


class _RoutingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        self.register_buffer('routing_bias', torch.tensor([0., 1 / 512], dtype=torch.float32))
        self.register_buffer('route_count', torch.tensor([2], dtype=torch.int64))


def _source(tmp_path):
    model = nn.Module()
    model.layers = nn.ModuleList([_RoutingLayer()])
    with torch.no_grad():
        model.layers[0].proj.weight.copy_(torch.eye(2))
    path = tmp_path / 'source.safetensors'
    save_file({name: value.contiguous() for name, value in model.state_dict().items()}, str(path))
    names = list(model.state_dict())
    model.to_empty(device='meta')
    return model, {name: str(path) for name in names}, {name: name for name in names}


def _assert_routing_precision(model):
    layer = model.layers[0]
    assert layer.proj.weight.dtype == torch.bfloat16
    assert layer.routing_bias.dtype == torch.float32
    assert layer.route_count.dtype == torch.int64
    # All these values are exactly representable in BF16. Narrowing the
    # BUFFER still changes arithmetic: the BF16 sum rounds away the bias.
    scores = torch.tensor([[0.5, 0.5]], dtype=torch.bfloat16)
    corrected = scores + layer.routing_bias
    assert corrected[0, 1] > corrected[0, 0]
    assert corrected.argmax(-1).item() == 1


def test_resident_materialization_preserves_declared_buffer_precision(tmp_path):
    model, shards, keys = _source(tmp_path)
    assert _materialize(model, ['layers.0.'], shards, keys, torch.device('cpu'), torch.bfloat16) == 3
    _assert_routing_precision(model)


@pytest.mark.parametrize('prefetch', [False, True])
def test_streaming_cache_preserves_declared_buffer_precision(tmp_path, prefetch):
    model, shards, keys = _source(tmp_path)
    pool = ThreadPoolExecutor(max_workers=1)
    context = StreamingContext(model=model, base_model=model, layers=model.layers,
        layers_prefix='layers.', num_layers=1,
        install_resolvers=[_build_install_resolver(model, 'layers.0')],
        weight_shard=shards, weight_ckpt=keys,
        layer_cache=LayerCache(max_bytes=1024), prefetch_pool=pool,
        device=torch.device('cpu'), dtype=torch.bfloat16, offload_folder=str(tmp_path))
    try:
        if prefetch:
            context.schedule_prefetch(0)
        context.install(0, require_prefetched=prefetch)
        _assert_routing_precision(model)
        context.unload(0)
        context.install(0, require_prefetched=True)
        _assert_routing_precision(model)
    finally:
        context.shutdown()
