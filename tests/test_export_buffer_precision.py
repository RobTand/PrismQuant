"""The compressed-tensors export must not narrow persistent FP32 buffers.

`_read_layer_to_device` casts every floating tensor to the requested
PARAMETER dtype (BF16 on the shipping recipe). Section 3e of
`materialize_tensors_streaming` then emits persistent layer buffers through
`_passthrough_tensor`, which restores the dtype the source declared but
cannot restore values the read already threw away. A router bias is the
concrete case: BF16 score + FP32 bias sums in FP32, BF16 score + BF16 bias
rounds before top-k, so the dtype is arithmetic and not just storage.

#310 / PR #312 gave the reader a `buffer_dtypes` map and threaded it through
the resident and streaming source loads. This file covers the export caller,
which is the remaining one -- both that the wiring is there (a missing kwarg
is silent, and the emitted checkpoint is what changes) and that the bytes
that come out the far end are the source bytes.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from prismaquant.export_native_compressed import (  # noqa: E402
    _build_source_dtype_map,
    _passthrough_tensor,
    materialize_tensors_streaming,
)
from prismaquant.layer_streaming import (  # noqa: E402
    _build_install_resolver,
    _fast_install,
    _read_layer_to_device,
)

# 0.5 and 0.5 + 2**-12 are two distinct FP32 values inside ONE BF16 ulp
# (2**-9 at 0.5), so the cast collapses them onto each other. Two experts
# whose biases differ by less than a BF16 ulp are exactly the case where
# preserving the buffer decides the routing: in FP32 the second wins, and
# after the narrowing they tie and top-1 falls back to the first.
_BIAS_LO = 0.5
_BIAS_HI = 0.5 + 2.0 ** -12


class _RoutingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        # LFM's `expert_bias` shape: FP32, persistent, added to the router
        # score before top-k.
        self.register_buffer(
            'expert_bias',
            torch.tensor([_BIAS_LO, _BIAS_HI], dtype=torch.float32))
        self.register_buffer(
            'route_count', torch.tensor([2], dtype=torch.int64))
        self.register_buffer(
            'derived', torch.zeros(2, dtype=torch.float32), persistent=False)


def _source_layer(tmp_path):
    """A meta skeleton plus the shard it streams from, shaped the way
    `materialize_tensors_streaming` has them at the layer loop."""
    model = nn.Module()
    model.layers = nn.ModuleList([_RoutingLayer()])
    with torch.no_grad():
        model.layers[0].proj.weight.copy_(torch.eye(2))
    path = tmp_path / 'source.safetensors'
    state = {name: value.contiguous()
             for name, value in model.state_dict().items()}
    save_file(state, str(path))
    names = list(state)
    model.to_empty(device='meta')
    shards = {name: str(path) for name in names}
    keys = {name: name for name in names}
    return model, shards, keys


def _emit_layer(tmp_path, *, buffer_dtypes):
    """Run the export's read -> install -> passthrough-emit sequence."""
    model, shards, keys = _source_layer(tmp_path)
    source_dtype_by_name = _build_source_dtype_map(shards, keys)
    tensors = _read_layer_to_device(
        'layers.0.', shards, keys, torch.bfloat16, torch.device('cpu'),
        buffer_dtypes=buffer_dtypes)
    resolver = _build_install_resolver(model, 'layers.0')
    _fast_install(resolver, tensors, torch.device('cpu'), model=model)

    layer_mod = model.get_submodule('layers.0')
    out = {}
    for mod_name, mod in layer_mod.named_modules():
        non_persistent = getattr(mod, '_non_persistent_buffers_set', set())
        for buf_name, buf in mod.named_buffers(recurse=False):
            if buf_name in non_persistent or buf.is_meta:
                continue
            full_modpath = f'layers.0.{mod_name}' if mod_name else 'layers.0'
            full = f'{full_modpath}.{buf_name}'
            out[full], _ = _passthrough_tensor(
                full, buf, source_dtype_by_name)
    for name, param in layer_mod.named_parameters():
        out[f'layers.0.{name}'] = param.detach()
    return out


def _declared(tmp_path):
    model, _, _ = _source_layer(tmp_path)
    return {name: value.dtype
            for name, value in model.named_buffers(remove_duplicate=False)}


def test_export_emits_source_buffer_bytes(tmp_path):
    """The FP32 router bias survives the export read bit-identically."""
    emitted = _emit_layer(tmp_path, buffer_dtypes=_declared(tmp_path))
    bias = emitted['layers.0.expert_bias']
    assert bias.dtype == torch.float32
    assert torch.equal(
        bias, torch.tensor([_BIAS_LO, _BIAS_HI], dtype=torch.float32))
    # Non-float buffers were never at risk and must stay untouched; the
    # non-persistent one is not emitted at all.
    assert emitted['layers.0.route_count'].dtype == torch.int64
    assert 'layers.0.derived' not in emitted


def test_export_still_narrows_parameters_to_the_recipe_dtype(tmp_path):
    """Buffer preservation must not widen the parameter policy."""
    emitted = _emit_layer(tmp_path, buffer_dtypes=_declared(tmp_path))
    assert emitted['layers.0.proj.weight'].dtype == torch.bfloat16


def test_a_narrowed_bias_changes_routing(tmp_path):
    """Why the loss is not cosmetic. The same emit run WITHOUT the declared
    map relabels the bias FP32 on the way out -- `_passthrough_tensor`
    restores the dtype -- but the two experts have already been collapsed
    onto one BF16 value, and the expert the source ranked first loses."""
    narrowed = _emit_layer(tmp_path, buffer_dtypes=None)['layers.0.expert_bias']
    preserved = _emit_layer(
        tmp_path, buffer_dtypes=_declared(tmp_path))['layers.0.expert_bias']
    # Both are FP32 on the wire: the dtype survives, the values do not.
    assert narrowed.dtype == preserved.dtype == torch.float32
    assert not torch.equal(narrowed, preserved)
    scores = torch.zeros(1, 2, dtype=torch.bfloat16)
    assert (scores + preserved).argmax(-1).item() == 1
    assert (scores + narrowed).argmax(-1).item() == 0


def test_export_layer_read_is_wired_to_the_declared_buffer_dtypes():
    """The wiring itself, because dropping the kwarg is silent: nothing
    raises, the histogram still labels the buffer FP32, and only the
    emitted values change. Guards the one line the behaviour above
    depends on inside a 7k-line function no unit test can drive."""
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(materialize_tensors_streaming)))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == '_read_layer_to_device']
    assert calls, 'export no longer reads layers through _read_layer_to_device'
    for call in calls:
        assert any(kw.arg == 'buffer_dtypes' for kw in call.keywords), (
            'materialize_tensors_streaming reads a layer without passing '
            'buffer_dtypes; persistent FP32 buffers narrow to the parameter '
            'dtype and _passthrough_tensor cannot recover them')
