import pytest
import torch
import torch.nn as nn

from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.weight_session import WeightSession


class _ModelWithBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.proj = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.model.proj(x)


def test_weight_session_accepts_language_model_alias_for_staged_body():
    model = _ModelWithBody().eval()
    session = WeightSession(model)

    session.initialize({"model.language_model.proj": "BF16"}, units=[])

    assert session.current_assignment() == {
        "model.language_model.proj": "BF16"
    }
    assert session.diagnostics()["n_bf16_snapshots"] == 0
    assert session.format_weight("model.language_model.proj", "BF16") is not None
    assert session.diagnostics()["n_bf16_snapshots"] == 1


def test_weight_session_rejects_strict_production_cache_miss():
    model = _ModelWithBody().eval()
    cache = ProductionWeightCache(weights={}, levers={})
    session = WeightSession(model, production_weight_cache=cache)

    with pytest.raises(RuntimeError, match="production_weight_cache miss"):
        session.initialize({"model.proj": "NVFP4"}, units=[])


def test_weight_session_allows_mxfp8_rtn_fallback_with_nvfp4_only_cache():
    model = _ModelWithBody().eval()
    cache = ProductionWeightCache(weights={}, levers={})
    session = WeightSession(model, production_weight_cache=cache)

    session.initialize({"model.proj": "MXFP8_E4M3"}, units=[])

    diag = session.diagnostics()
    assert diag["n_rtn_fallbacks"] == 1
    assert diag["n_cache_misses"] == 0


def test_weight_session_reuses_existing_spilled_snapshot(tmp_path):
    model = _ModelWithBody().eval()
    model.model.proj.weight.data.fill_(1.0)
    first = WeightSession(model, snapshot_dir=str(tmp_path))
    saved = first.format_weight("model.proj", "BF16")

    model.model.proj.weight.data.zero_()
    second = WeightSession(model, snapshot_dir=str(tmp_path))
    reused = second.format_weight("model.proj", "BF16")

    assert saved is not None
    assert reused is not None
    torch.testing.assert_close(reused, torch.ones_like(reused))
    assert second.diagnostics()["n_bf16_snapshots"] == 1
