"""Linear-attention (recurrent) hybrid masking and layer-type resolution.

Pins the PR #80 review contract. Two halves:

- `linear_attention` layers are routed through the CURRENT (transformers
  >= 5.15) recurrent-mask contract, always via the local
  `_recurrent_padding_mask` shim — never the upstream helper, whose
  5.13/5.14 incarnation still has the pre-fix cache-state contract:
  2D padding mask trimmed to the local sequence, `None` whenever masking
  would be a no-op — never the dense additive causal mask, never the raw
  un-trimmed growing mask, and identical semantics with or without a
  populated cache.
- layer-type resolution covers the transformers>=5.13 `.block_type` rename
  and the Qwen3.5/3.6 `linear_attn` child module through the existing
  attribute/index/config lookups, and an otherwise unknown layer still
  fails closed instead of being guessed structurally.

No transformers modeling dependency — fakes only, same as
test_multilayer_rope_forward.py.
"""
import pytest
import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from prismaquant.layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _layer_attention_type,
    _recurrent_padding_mask,
)


# --- fakes -----------------------------------------------------------------
class _Base(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config


class _RecorderLayer(nn.Module):
    """Records the attention_mask it actually received."""
    def __init__(self, *, block_type=None, layer_type=None):
        super().__init__()
        if block_type is not None:
            self.block_type = block_type
        if layer_type is not None:
            self.layer_type = layer_type
        self.received_mask = "UNSET"

    def forward(self, *, hidden_states, **kw):
        self.received_mask = kw.get("attention_mask")
        return hidden_states


def _hybrid_cfg():
    cfg = PreTrainedConfig()
    cfg.is_causal = True
    cfg.layer_types = ["full_attention", "linear_attention"]
    cfg._attn_implementation = "eager"
    return cfg


# --- layer-type resolution -------------------------------------------------
@pytest.mark.parametrize("lt", ["linear_attention", "full_attention"])
def test_block_type_resolves(lt):
    # transformers>=5.13: hybrid decoder layers store `.block_type`
    layer = nn.Module()
    layer.block_type = lt
    assert _layer_attention_type(layer) == lt


@pytest.mark.parametrize("lt", ["linear_attention", "sliding_attention"])
def test_legacy_layer_type_still_resolves(lt):
    # transformers<=5.12 name — must keep working unchanged
    layer = nn.Module()
    layer.layer_type = lt
    assert _layer_attention_type(layer) == lt


def test_linear_attn_child_layer_type_resolves():
    # Qwen3_5GatedDeltaNet carries its own `layer_type`; the outer layer
    # exposes neither layer_type/block_type nor self_attn/attention.
    layer = nn.Module()
    layer.linear_attn = nn.Module()
    layer.linear_attn.layer_type = "linear_attention"
    assert _layer_attention_type(layer) == "linear_attention"


def test_linear_attn_child_layer_idx_config_resolves():
    # ...and its `layer_idx` feeds the generic config.layer_types fallback.
    layer = nn.Module()
    layer.linear_attn = nn.Module()
    layer.linear_attn.layer_idx = 1
    layer.config = _hybrid_cfg()
    assert _layer_attention_type(layer) == "linear_attention"


def _lfm_cfg():
    # LFM2.5's schedule: a short-convolution mixer on most layers, GQA on the
    # rest. `_compute_attention_mask` already keys its dict on these strings.
    cfg = PreTrainedConfig()
    cfg.is_causal = True
    cfg.layer_types = ["conv", "full_attention"]
    cfg._attn_implementation = "eager"
    return cfg


def test_conv_mixer_child_names_an_lfm_layer():
    # RobTand/prismaquant#276: Lfm2MoeDecoderLayer carries no layer_type and
    # no attention module on a conv layer -- only `self.conv`, which holds
    # the layer_idx and the config the generic fallback needs.
    layer = nn.Module()
    layer.conv = nn.Module()
    layer.conv.layer_idx = 0
    layer.conv.config = _lfm_cfg()
    assert _layer_attention_type(layer) == "conv"


def test_lfm_conv_and_attention_layers_get_different_mask_entries():
    # The whole point of naming the layer: the conv layer must receive the
    # recurrent padding entry and the attention layer the dense causal one.
    cfg = _lfm_cfg()
    conv_layer = _RecorderLayer()
    conv_layer.conv = nn.Module()
    conv_layer.conv.layer_idx = 0
    conv_layer.conv.config = cfg
    attn_layer = _RecorderLayer()
    attn_layer.self_attn = nn.Module()
    attn_layer.self_attn.layer_idx = 1
    attn_layer.self_attn.config = cfg
    masks = {"full_attention": torch.zeros(1, 1, 2, 2), "conv": torch.ones(1, 2)}
    hidden = torch.zeros(1, 2, 4)
    _call_layer(conv_layer, hidden, position_embeddings=None,
                attention_mask=masks, position_ids=None)
    _call_layer(attn_layer, hidden, position_embeddings=None,
                attention_mask=masks, position_ids=None)
    assert conv_layer.received_mask is masks["conv"]
    assert attn_layer.received_mask is masks["full_attention"]


def test_unknown_self_attn_layer_fails_closed():
    # A layer whose type cannot be resolved must stay unresolved (None) and
    # make _call_layer raise — never silently assume full_attention.
    layer = _RecorderLayer()
    layer.self_attn = nn.Module()
    assert _layer_attention_type(layer) is None
    with pytest.raises(RuntimeError, match="known layer_type"):
        _call_layer(layer, torch.zeros(1, 2, 4),
                    position_embeddings=None,
                    attention_mask={"full_attention": None},
                    position_ids=None)


# --- recurrent-mask contract (direct) --------------------------------------
def test_continuation_mask_trims_to_local_sequence():
    hidden = torch.zeros(2, 4, 8)
    # growing cache-continuation mask: 6 total positions, local seq is 4
    pad = torch.tensor([[1, 1, 0, 1, 1, 1],
                        [0, 0, 1, 1, 1, 1]])
    out = _recurrent_padding_mask(hidden, pad)
    assert out.shape == (2, 4)
    assert torch.equal(out, pad[:, -4:])
    assert out.is_contiguous()


def test_single_token_decode_returns_none():
    hidden = torch.zeros(1, 1, 8)
    # decode step continuing a cached, left-padded prompt: not all-ones,
    # longer than the local sequence — None purely because local seq == 1
    pad = torch.tensor([[0, 1, 1, 1, 1, 1]])
    assert _recurrent_padding_mask(hidden, pad) is None


def test_non_2d_mask_returns_none():
    hidden = torch.zeros(1, 4, 8)
    pad4d = torch.zeros(1, 1, 4, 4)
    assert _recurrent_padding_mask(hidden, pad4d) is None


def test_missing_mask_returns_none():
    hidden = torch.zeros(1, 4, 8)
    assert _recurrent_padding_mask(hidden, None) is None


# --- routing through _compute_attention_mask -------------------------------
def test_left_padded_mask_routes_2d_to_linear():
    cfg = _hybrid_cfg()
    base = _Base(cfg)
    hidden = torch.zeros(1, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0)
    pad = torch.tensor([[0, 1, 1, 1]])

    masks = _compute_attention_mask(base, hidden, position_ids,
                                    attention_mask=pad)

    lin = masks["linear_attention"]
    assert lin is not None and lin.ndim == 2  # never the dense 4D mask
    assert torch.equal(lin, pad)
    assert lin.is_contiguous()
    assert masks["full_attention"].shape == (1, 1, 4, 4)


def test_all_ones_mask_routes_none_to_linear():
    cfg = _hybrid_cfg()
    base = _Base(cfg)
    hidden = torch.zeros(1, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0)

    masks = _compute_attention_mask(base, hidden, position_ids,
                                    attention_mask=torch.ones(1, 4))

    assert masks["linear_attention"] is None
    assert masks["full_attention"].shape == (1, 1, 4, 4)


def test_sliding_full_mapping_non_regression():
    # Gemma-style hybrid must be untouched by the linear-attention branch.
    cfg = PreTrainedConfig()
    cfg.is_causal = True
    cfg.layer_types = ["sliding_attention", "full_attention"]
    cfg.sliding_window = 2
    cfg._attn_implementation = "eager"
    base = _Base(cfg)
    hidden = torch.zeros(1, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0)

    masks = _compute_attention_mask(base, hidden, position_ids)

    assert set(masks) == {"sliding_attention", "full_attention"}
    assert masks["full_attention"].shape == (1, 1, 4, 4)
    assert masks["sliding_attention"].shape == (1, 1, 4, 4)
    assert float(masks["full_attention"][0, 0, 3, 0]) == 0.0
    assert float(masks["sliding_attention"][0, 0, 3, 0]) < -1e20


# --- end-to-end selection through _call_layer ------------------------------
def test_call_layer_delivers_recurrent_mask_to_linear_layer():
    pad = torch.tensor([[0, 1, 1, 1]])
    dense = torch.zeros(1, 1, 4, 4)
    masks = {"full_attention": dense, "linear_attention": pad}

    linear = _RecorderLayer(block_type="linear_attention")
    _call_layer(linear, torch.zeros(1, 4, 8), position_embeddings=None,
                attention_mask=masks, position_ids=None)
    assert linear.received_mask is pad

    full = _RecorderLayer(block_type="full_attention")
    _call_layer(full, torch.zeros(1, 4, 8), position_embeddings=None,
                attention_mask=masks, position_ids=None)
    assert full.received_mask is dense


# --- shim-only guarantee (5.13/5.14 old-helper regression) ------------------
def test_routing_never_consults_upstream_helper(monkeypatch):
    # transformers 5.13/5.14 ship create_recurrent_attention_mask with the
    # pre-fix contract (None whenever the cache has previous state, no
    # single-token case). The routing must therefore NEVER consult the
    # upstream helper, on any version — booby-trap it and exercise both
    # semantics Rob's review names: padded multi-token and single-token.
    try:
        import transformers.masking_utils as mu
    except ImportError:  # pragma: no cover
        mu = None
    if mu is not None and hasattr(mu, "create_recurrent_attention_mask"):
        def _trap(**kwargs):
            pytest.fail("upstream create_recurrent_attention_mask must not "
                        "be consulted (5.13/5.14 ship the pre-fix contract)")
        monkeypatch.setattr(mu, "create_recurrent_attention_mask", _trap)

    cfg = _hybrid_cfg()
    base = _Base(cfg)

    # padded multi-token: 2D mask kept and (trivially) trimmed, NOT None
    hidden = torch.zeros(1, 4, 8)
    pad = torch.tensor([[0, 1, 1, 1]])
    masks = _compute_attention_mask(base, hidden, torch.arange(4).unsqueeze(0),
                                    attention_mask=pad)
    assert torch.equal(masks["linear_attention"], pad)

    # single-token step: None
    hidden1 = torch.zeros(1, 1, 8)
    masks1 = _compute_attention_mask(base, hidden1, torch.zeros(1, 1,
                                                                dtype=torch.long),
                                     attention_mask=torch.ones(1, 1))
    assert masks1["linear_attention"] is None


def test_cached_continuation_contract_is_cache_state_independent():
    # The old 5.13/5.14 helper returned None whenever
    # past_key_values.has_previous_state() — silently dropping padding from
    # a multi-token cached continuation. The current contract (and our
    # shim, which takes no cache argument at all — by design) must keep and
    # trim the mask regardless of cache state.
    hidden = torch.zeros(2, 4, 8)
    continuation = torch.tensor([[1, 1, 0, 1, 1, 1],
                                 [0, 0, 1, 1, 1, 1]])  # 6 cached+new, 4 local
    out = _recurrent_padding_mask(hidden, continuation)
    assert out is not None and torch.equal(out, continuation[:, -4:])


# --- non-attention blocks in a mixed hybrid schedule ------------------------
def test_mixed_schedule_non_attention_blocks_get_none():
    # Nemotron-H-style schedule: moe/mlp blocks consume no attention mask —
    # upstream dispatches them None via `.get(block_type)`. An unknown
    # *attention* type still fails closed.
    cfg = PreTrainedConfig()
    cfg.is_causal = True
    cfg.layer_types = ["linear_attention", "moe", "full_attention", "mlp"]
    cfg._attn_implementation = "eager"
    base = _Base(cfg)
    hidden = torch.zeros(1, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0)
    pad = torch.tensor([[0, 1, 1, 1]])

    masks = _compute_attention_mask(base, hidden, position_ids,
                                    attention_mask=pad)

    assert masks["moe"] is None and masks["mlp"] is None
    assert torch.equal(masks["linear_attention"], pad)
    assert masks["full_attention"].shape == (1, 1, 4, 4)

    moe = _RecorderLayer(block_type="moe")
    _call_layer(moe, hidden, position_embeddings=None,
                attention_mask=masks, position_ids=position_ids)
    assert moe.received_mask is None

    # a declared but unbuildable attention type stays fail-closed
    cfg2 = PreTrainedConfig()
    cfg2.is_causal = True
    cfg2.layer_types = ["linear_attention", "quantum_attention",
                        "full_attention"]
    cfg2._attn_implementation = "eager"
    masks2 = _compute_attention_mask(_Base(cfg2), hidden, position_ids,
                                     attention_mask=pad)
    assert "quantum_attention" not in masks2
    unknown = _RecorderLayer(block_type="quantum_attention")
    with pytest.raises(RuntimeError, match="known layer_type"):
        _call_layer(unknown, hidden, position_embeddings=None,
                    attention_mask=masks2, position_ids=position_ids)


def test_nonattention_schedule_without_recurrent_layers_takes_dict_path():
    # Self-found adjacent gap: a schedule declaring moe/mlp WITHOUT any
    # linear/sliding/conv layer must still take the per-type dict path —
    # the single-dense-mask early return would otherwise feed the dense
    # mask to the non-attention blocks too.
    cfg = PreTrainedConfig()
    cfg.is_causal = True
    cfg.layer_types = ["full_attention", "moe", "mlp"]
    cfg._attn_implementation = "eager"
    hidden = torch.zeros(1, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0)

    masks = _compute_attention_mask(_Base(cfg), hidden, position_ids)

    assert isinstance(masks, dict)
    assert masks["moe"] is None and masks["mlp"] is None
    assert masks["full_attention"].shape == (1, 1, 4, 4)

    # non-regression: a plain all-full schedule keeps the bare dense mask
    cfg2 = PreTrainedConfig()
    cfg2.is_causal = True
    cfg2.layer_types = ["full_attention", "full_attention"]
    cfg2._attn_implementation = "eager"
    bare = _compute_attention_mask(_Base(cfg2), hidden, position_ids)
    assert not isinstance(bare, dict) and bare.shape == (1, 1, 4, 4)
