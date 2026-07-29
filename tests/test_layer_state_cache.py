from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant.layer_state_cache import LayerHiddenStateCache
from prismaquant.production_weight_cache import ProductionWeightCache


class _LinearBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class _TinyDecoder(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_LinearBlock(hidden) for _ in range(layers)])
        self.norm = nn.Identity()


class _TinyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocab: int = 17,
        hidden: int = 8,
        layers: int = 4,
        use_lm_head: bool = True,
    ):
        super().__init__()
        self.model = _TinyDecoder(vocab, hidden, layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False) if use_lm_head else nn.Identity()

    def forward(self, input_ids):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        hidden = self.model.norm(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden), last_hidden_state=hidden)


class _CountingHead(nn.Linear):
    def __init__(self, hidden: int, vocab: int):
        super().__init__(hidden, vocab, bias=False)
        self.last_input_shape = None

    def forward(self, input):  # noqa: A002 - mirrors torch.nn.Module API
        self.last_input_shape = tuple(input.shape)
        return super().forward(input)


class _MemoryDecoder(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Identity() for _ in range(layers)])
        self.norm = nn.Identity()


class _MemoryModel(nn.Module):
    def __init__(self, *, vocab: int = 31, hidden: int = 16, layers: int = 36):
        super().__init__()
        self.model = _MemoryDecoder(vocab, hidden, layers)
        self.lm_head = nn.Identity()

    def forward(self, input_ids):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        hidden = self.model.norm(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden), last_hidden_state=hidden)


def _calib_ids(batch: int = 3, seq: int = 5, vocab: int = 17) -> torch.Tensor:
    return torch.arange(batch * seq, dtype=torch.long).reshape(batch, seq) % vocab


def test_layer_state_cache_replay_from_zero_matches_full_forward():
    torch.manual_seed(0)
    model = _TinyCausalLM().eval()
    calib_ids = _calib_ids()
    cache = LayerHiddenStateCache(model)

    cache.populate({}, calib_ids, device="cpu", dtype=torch.float32)

    full_logits = model(calib_ids).logits
    replay_logits = cache.replay_from(0)
    torch.testing.assert_close(replay_logits, full_logits, rtol=0, atol=0)


def test_layer_state_cache_replay_from_mid_with_no_override():
    torch.manual_seed(1)
    model = _TinyCausalLM(layers=5).eval()
    calib_ids = _calib_ids(batch=2, seq=4)
    cache = LayerHiddenStateCache(model)

    cache.populate({}, calib_ids, device="cpu", dtype=torch.float32)

    full_logits = model(calib_ids).logits
    for layer_idx in (1, 2, 4):
        replay_logits = cache.replay_from(layer_idx)
        torch.testing.assert_close(replay_logits, full_logits, rtol=0, atol=0)


def test_layer_state_cache_last_token_logits_skips_full_sequence_lm_head():
    torch.manual_seed(11)
    model = _TinyCausalLM(layers=5).eval()
    hidden = model.lm_head.in_features
    vocab = model.lm_head.out_features
    model.lm_head = _CountingHead(hidden, vocab)
    calib_ids = _calib_ids(batch=2, seq=7)
    cache = LayerHiddenStateCache(model)

    cache.populate({}, calib_ids, device="cpu", dtype=torch.float32)

    full_logits = model(calib_ids).logits
    last_logits = cache.replay_from(2, last_token_only=True)

    # Bit-identical on the reference box, where both calls hit the same kernel.
    # On CPU the BLAS dispatches a different matmul for (B,1,H)@(H,V) than for
    # (B,S,H)@(H,V), so fp32 accumulation order differs by ~4e-09. The claim
    # under test is that the shortcut returns the SAME logits while skipping
    # the full-sequence lm_head -- the shape assertion below is the load-bearing
    # half; exact bit equality is a property of the BLAS, not of this code.
    torch.testing.assert_close(last_logits, full_logits[:, -1:, :],
                               rtol=1e-6, atol=1e-6)
    assert model.lm_head.last_input_shape == (2, 1, hidden)


def test_layer_state_cache_replay_with_weight_override():
    torch.manual_seed(2)
    model = _TinyCausalLM(layers=4, use_lm_head=False).eval()
    calib_ids = _calib_ids(batch=2, seq=3)
    cache = LayerHiddenStateCache(model)
    cache.populate({}, calib_ids, device="cpu", dtype=torch.float32)

    layer_idx = 3
    target = model.model.layers[layer_idx].proj
    target_name = f"model.layers.{layer_idx}.proj"
    original_weight = target.weight.detach().clone()
    baseline = cache.replay_from(layer_idx)
    zero_weight = torch.zeros_like(target.weight)
    overridden = cache.replay_from(layer_idx, {target_name: zero_weight})
    contribution = target(cache.layer_inputs[layer_idx])

    assert not torch.equal(baseline, overridden)
    torch.testing.assert_close(baseline - overridden, contribution, rtol=0, atol=0)
    torch.testing.assert_close(target.weight, original_weight, rtol=0, atol=0)


def test_layer_state_cache_external_weight_management_skips_baseline_weight_clones(monkeypatch):
    torch.manual_seed(12)
    model = _TinyCausalLM(layers=2, use_lm_head=False).eval()
    calib_ids = _calib_ids(batch=1, seq=3)
    cache = LayerHiddenStateCache(model)

    monkeypatch.setenv("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
    cache.populate(
        {"model.layers.0.proj": "BF16", "model.layers.1.proj": "BF16"},
        calib_ids,
        device="cpu",
        dtype=torch.float32,
    )

    assert cache._baseline_weight_values == {}
    replay_logits = cache.replay_from(0)
    torch.testing.assert_close(replay_logits, model(calib_ids).logits, rtol=0, atol=0)


def test_layer_state_cache_production_cache_miss_is_strict_by_default(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_STRICT_PRODUCTION_CACHE", raising=False)
    model = _TinyCausalLM(layers=2, use_lm_head=False).eval()
    calib_ids = _calib_ids(batch=1, seq=3)
    cache = LayerHiddenStateCache(model)

    with pytest.raises(RuntimeError, match="production_weight_cache miss"):
        cache.populate(
            {"model.layers.0.proj": "NVFP4"},
            calib_ids,
            device="cpu",
            dtype=torch.float32,
            production_weight_cache=ProductionWeightCache({}, levers={}),
        )


def test_layer_state_cache_strict_miss_escape_allows_rtn(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_STRICT_PRODUCTION_CACHE", "0")
    model = _TinyCausalLM(layers=2, use_lm_head=False).eval()
    calib_ids = _calib_ids(batch=1, seq=3)
    cache = LayerHiddenStateCache(model)

    cache.populate(
        {"model.layers.0.proj": "NVFP4"},
        calib_ids,
        device="cpu",
        dtype=torch.float32,
        production_weight_cache=ProductionWeightCache({}, levers={}),
    )

    assert len(cache.layer_inputs) == 2


def test_layer_state_cache_invalidate_clears_state():
    torch.manual_seed(3)
    model = _TinyCausalLM().eval()
    calib_ids = _calib_ids()
    cache = LayerHiddenStateCache(model)
    cache.populate({}, calib_ids, device="cpu", dtype=torch.float32)

    cache.invalidate()

    assert cache.layer_inputs == []
    with pytest.raises(RuntimeError, match=r"call populate\(\.\.\.\) before replay_from"):
        cache.replay_from(0)


def test_layer_state_cache_memory_bounded():
    torch.manual_seed(4)
    batch, seq, hidden, layers = 2, 8, 16, 36
    model = _MemoryModel(hidden=hidden, layers=layers).eval()
    calib_ids = _calib_ids(batch=batch, seq=seq, vocab=31)
    cache = LayerHiddenStateCache(model)

    cache.populate({}, calib_ids, device="cpu", dtype=torch.bfloat16)

    expected = layers * batch * seq * hidden * torch.empty((), dtype=torch.bfloat16).element_size()
    actual = cache.cache_nbytes()
    assert len(cache.layer_inputs) == layers
    assert abs(actual - expected) / expected <= 0.10
