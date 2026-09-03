"""Per-layer streaming + per-pair fit render identity (#172, #173).

#172: streaming packed appends (``module_acts_override``) record no
render-identity section, so a hand-mixed streaming build merges silently.
The fix records a per-layer section list keyed by the module each call saw.

#173: same-directory resident packed appends with different fit calibs merge
silently for NEW pairs. The fix binds the fit calibration per pair, so the
M4 lazy gap-fill (disjoint pairs, render-split calib) records its own split
while a same-pair conflict refuses.
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

import prismaquant.production_weight_cache as pwc
from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    fill_packed_expert_cache_entries,
)

PACKED_APPEND_SIDECAR_KEY = getattr(
    pwc, "PACKED_APPEND_SIDECAR_KEY", "packed_expert_append"
)
STREAMING_APPEND_SIDECAR_KEY = getattr(
    pwc, "PACKED_STREAMING_APPEND_SIDECAR_KEY",
    "packed_expert_streaming_append",
)

from test_packed_expert_cross_domain_gate import (
    TinyLM,
    TinyMlp,
)


def _packed_cache(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ProductionWeightCache(
        weights={},
        levers={"gptq": True},
        activation_max_abs={},
        failed={},
        cache_dir=str(cache_dir),
        metadata={},
    )


def _sidecar(cache_dir):
    return cache_dir / pwc.RENDER_IDENTITY_SIDECAR_FILENAME


def _streaming_append(cache, model, module_acts, cache_dir, **kwargs):
    params = {
        "render_assignment": {
            "mlp.experts.gate_up_proj": "NVFP4",
            "mlp.experts.down_proj": "NVFP4",
        },
        "levers": {"gptq": True},
        "profile": None,
        "module_token_budget": 4096,
        "cache_dir": cache_dir,
        "progress": False,
        "module_acts_override": module_acts,
    }
    params.update(kwargs)
    return fill_packed_expert_cache_entries(
        cache, model, None, **params
    )


def _snapshot(rows=64, hidden=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=g)


# ---------------------------------------------------------------------------
# #172: streaming per-layer section list
# ---------------------------------------------------------------------------

def test_streaming_append_records_per_layer_section(tmp_path):
    torch.manual_seed(11)
    model = TinyLM().eval()
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)

    _streaming_append(
        cache, model, {"mlp.experts": _snapshot()}, cache_dir
    )

    sidecar = json.loads(_sidecar(cache_dir).read_text())
    assert STREAMING_APPEND_SIDECAR_KEY in sidecar
    layers = sidecar[STREAMING_APPEND_SIDECAR_KEY]["layers"]
    assert set(layers) == {"mlp.experts"}


def test_streaming_append_refuses_different_budget_for_same_module(tmp_path):
    torch.manual_seed(11)
    model = TinyLM().eval()
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)

    _streaming_append(
        cache, model, {"mlp.experts": _snapshot()}, cache_dir,
        module_token_budget=4096,
    )

    with pytest.raises(
        ValueError, match="module_token_budget"
    ) as exc_info:
        _streaming_append(
            cache, model, {"mlp.experts": _snapshot()}, cache_dir,
            module_token_budget=512,
        )
    assert "instead of appending to it" in str(exc_info.value)


def test_streaming_append_refuses_different_acts_for_same_module(tmp_path):
    torch.manual_seed(11)
    model = TinyLM().eval()
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)

    _streaming_append(
        cache, model, {"mlp.experts": _snapshot(seed=0)}, cache_dir
    )

    with pytest.raises(ValueError, match="mlp.experts") as exc_info:
        _streaming_append(
            cache, model, {"mlp.experts": _snapshot(seed=99)}, cache_dir
        )
    assert "instead of appending to it" in str(exc_info.value)


class _TwoLayerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = TinyMlp()
        self.layer1 = TinyMlp()

    def forward(self, input_ids, use_cache=False):  # pragma: no cover
        raise NotImplementedError("streaming tests supply module acts")


def _two_layer_streaming_append(
    cache, model, module_acts, assignment, cache_dir, **kwargs
):
    params = {
        "levers": {"gptq": True},
        "profile": None,
        "module_token_budget": 4096,
        "cache_dir": cache_dir,
        "progress": False,
        "module_acts_override": module_acts,
        "render_assignment": assignment,
    }
    params.update(kwargs)
    return fill_packed_expert_cache_entries(
        cache, model, None, **params
    )


def test_streaming_append_merges_two_layers(tmp_path):
    torch.manual_seed(11)
    model = _TwoLayerLM().eval()
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)
    assignment = {
        "layer0.experts.gate_up_proj": "NVFP4",
        "layer0.experts.down_proj": "NVFP4",
        "layer1.experts.gate_up_proj": "NVFP4",
        "layer1.experts.down_proj": "NVFP4",
    }

    _two_layer_streaming_append(
        cache, model, {"layer0.experts": _snapshot(seed=0)},
        assignment, cache_dir,
    )
    _two_layer_streaming_append(
        cache, model, {"layer1.experts": _snapshot(seed=1)},
        assignment, cache_dir,
    )

    sidecar = json.loads(_sidecar(cache_dir).read_text())
    layers = sidecar[STREAMING_APPEND_SIDECAR_KEY]["layers"]
    assert set(layers) == {"layer0.experts", "layer1.experts"}


# ---------------------------------------------------------------------------
# #173: per-pair fit calibration binding
# ---------------------------------------------------------------------------

def _resident_append(cache, model, calib, assignment, cache_dir, **kwargs):
    params = {
        "render_assignment": dict(assignment),
        "levers": {"gptq": True},
        "profile": None,
        "module_token_budget": 4096,
        "eval_rows_per_expert": 8,
        "cache_dir": cache_dir,
        "progress": False,
    }
    params.update(kwargs)
    return fill_packed_expert_cache_entries(cache, model, calib, **params)


def test_packed_append_records_per_pair_fit_hash(tmp_path):
    torch.manual_seed(11)
    model = TinyLM().eval()
    calib = torch.randint(0, 32, (2, 64))
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)
    assignment = {
        "mlp.experts.gate_up_proj": "NVFP4",
        "mlp.experts.down_proj": "NVFP4",
    }

    _resident_append(cache, model, calib, assignment, cache_dir)

    from prismaquant.perturbed_x_cache import calibration_data_hash

    sidecar = json.loads(_sidecar(cache_dir).read_text())
    pair_map = sidecar[PACKED_APPEND_SIDECAR_KEY][
        "pair_fit_calibration_hashes"
    ]
    expected = calibration_data_hash(calib)
    assert pair_map == {
        "mlp.experts.gate_up_proj|NVFP4": expected,
        "mlp.experts.down_proj|NVFP4": expected,
    }


def test_packed_append_disjoint_pairs_different_calib_merge(tmp_path):
    """The M4 gap-fill pattern: disjoint pairs under a different calib merge,
    each keeping its own fit hash."""
    torch.manual_seed(11)
    model = TinyLM().eval()
    first_calib = torch.randint(0, 32, (2, 64))
    second_calib = torch.randint(0, 32, (1, 40))
    assert not torch.equal(first_calib, second_calib)
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)

    _resident_append(
        cache, model, first_calib,
        {"mlp.experts.gate_up_proj": "NVFP4"}, cache_dir,
    )
    _resident_append(
        cache, model, second_calib,
        {"mlp.experts.down_proj": "NVFP4"}, cache_dir,
    )

    from prismaquant.perturbed_x_cache import calibration_data_hash

    sidecar = json.loads(_sidecar(cache_dir).read_text())
    pair_map = sidecar[PACKED_APPEND_SIDECAR_KEY][
        "pair_fit_calibration_hashes"
    ]
    assert pair_map == {
        "mlp.experts.gate_up_proj|NVFP4":
            calibration_data_hash(first_calib),
        "mlp.experts.down_proj|NVFP4":
            calibration_data_hash(second_calib),
    }
    assert cache.resolve_key("mlp.experts.gate_up_proj", "NVFP4") is not None
    assert cache.resolve_key("mlp.experts.down_proj", "NVFP4") is not None


def test_packed_append_same_pair_different_calib_refuses(tmp_path):
    torch.manual_seed(11)
    model = TinyLM().eval()
    first_calib = torch.randint(0, 32, (2, 64))
    second_calib = torch.randint(0, 32, (1, 40))
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)
    assignment = {"mlp.experts.gate_up_proj": "NVFP4"}

    _resident_append(cache, model, first_calib, assignment, cache_dir)

    with pytest.raises(ValueError, match="pair_fit") as exc_info:
        _resident_append(cache, model, second_calib, assignment, cache_dir)
    assert "instead of appending to it" in str(exc_info.value)


def test_base_fill_adopts_streaming_sidecar(tmp_path, monkeypatch):
    """A base fill after a streaming append keeps the streaming section.

    The streaming append writes an append-only sidecar (no base fields);
    the base fill adopts it the same way it adopts the resident sections.
    """
    from test_production_cache_render_identity import (
        _TinyTwoLinear,
        _fake_render_env,
        _fill,
    )

    torch.manual_seed(11)
    model = TinyLM().eval()
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)

    _streaming_append(
        cache, model, {"mlp.experts": _snapshot()}, cache_dir
    )
    assert STREAMING_APPEND_SIDECAR_KEY in json.loads(
        _sidecar(cache_dir).read_text()
    )

    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, cache_dir)

    merged = json.loads(_sidecar(cache_dir).read_text())
    assert merged["render_scope"]
    assert set(
        merged[STREAMING_APPEND_SIDECAR_KEY]["layers"]
    ) == {"mlp.experts"}
