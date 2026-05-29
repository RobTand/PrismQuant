"""Generic per-expert -> packed-3D install bridge.

Some MoE checkpoints ship each routed expert's projections separately on
disk (``…experts.{i}.{proj}.weight``) while the live transformers module
exposes one packed ``[num_experts, …]`` parameter per projection group.
`_pack_per_expert_into_packed` stacks the former into the latter, driven
entirely by the model profile's packed-experts spec — no architecture
names in the loader. These tests pin the exact layout (against the
profile that motivated it, LFM2.5) and the safety behaviors.
"""
import re

import torch

from prismaquant.layer_streaming import _pack_per_expert_into_packed
from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile


def _lfm_pat():
    prof = Lfm2MoeProfile()
    regex = prof.per_expert_moe_regex()
    pat = re.compile(regex[len("re:"):] if regex.startswith("re:") else regex)
    return prof, pat


def _pack(out, prof, pat, live_shapes):
    return _pack_per_expert_into_packed(
        out,
        per_expert_re=pat,
        parent_for_projection=prof.packed_expert_parent_for_projection,
        projection_names_for=prof.packed_expert_projection_names,
        live_param_shape=live_shapes.get,
    )


def test_lfm_layout_exact():
    """gate_up_proj[i] == cat([w1_i, w3_i], dim=0); down_proj[i] == w2_i;
    experts stacked on a leading axis in index order."""
    prof, pat = _lfm_pat()
    E, I, H = 4, 6, 8
    blk = "model.layers.2.feed_forward.experts"
    out, w1s, w3s, w2s = {}, {}, {}, {}
    for i in range(E):
        w1, w3, w2 = torch.randn(I, H), torch.randn(I, H), torch.randn(H, I)
        out[f"{blk}.{i}.w1.weight"] = w1
        out[f"{blk}.{i}.w3.weight"] = w3
        out[f"{blk}.{i}.w2.weight"] = w2
        w1s[i], w3s[i], w2s[i] = w1, w3, w2
    out["model.layers.2.feed_forward.gate.weight"] = torch.randn(E, H)  # bystander

    live = {f"{blk}.gate_up_proj": (E, 2 * I, H), f"{blk}.down_proj": (E, H, I)}
    n = _pack(out, prof, pat, live)

    assert n == 2
    gup, dwn = out[f"{blk}.gate_up_proj"], out[f"{blk}.down_proj"]
    assert tuple(gup.shape) == (E, 2 * I, H)
    assert tuple(dwn.shape) == (E, H, I)
    for i in range(E):
        assert torch.equal(gup[i], torch.cat([w1s[i], w3s[i]], dim=0))
        assert torch.equal(dwn[i], w2s[i])
    # per-expert keys consumed, unrelated key preserved
    assert all(f"{blk}.{i}.w1.weight" not in out for i in range(E))
    assert "model.layers.2.feed_forward.gate.weight" in out


def test_noop_when_live_not_packed():
    """A per-expert *live* layout (no packed param) must be left untouched."""
    prof, pat = _lfm_pat()
    blk = "model.layers.2.feed_forward.experts"
    out = {f"{blk}.0.w1.weight": torch.randn(6, 8)}
    n = _pack(out, prof, pat, live_shapes={})  # nothing reports a packed shape
    assert n == 0
    assert f"{blk}.0.w1.weight" in out


def test_noop_when_no_per_expert_keys():
    """Already-packed checkpoint (no per-expert keys) is a clean no-op."""
    prof, pat = _lfm_pat()
    blk = "model.layers.2.feed_forward.experts"
    out = {f"{blk}.gate_up_proj": torch.randn(4, 12, 8)}
    n = _pack(out, prof, pat, live_shapes={f"{blk}.gate_up_proj": (4, 12, 8)})
    assert n == 0
    assert tuple(out[f"{blk}.gate_up_proj"].shape) == (4, 12, 8)


def test_shape_mismatch_fails_loud():
    """A wrong live shape must raise, never silently mis-pack."""
    prof, pat = _lfm_pat()
    E, I, H = 4, 6, 8
    blk = "model.layers.2.feed_forward.experts"
    out = {}
    for i in range(E):
        out[f"{blk}.{i}.w1.weight"] = torch.randn(I, H)
        out[f"{blk}.{i}.w3.weight"] = torch.randn(I, H)
        out[f"{blk}.{i}.w2.weight"] = torch.randn(H, I)
    bad = {f"{blk}.gate_up_proj": (E, 999, H), f"{blk}.down_proj": (E, H, I)}
    try:
        _pack(out, prof, pat, bad)
    except ValueError as e:
        assert "shape" in str(e)
    else:
        raise AssertionError("expected ValueError on shape mismatch")


def test_short_conv_linears_pinned_for_vllm():
    """vLLM builds the LFM2.5 short-conv mixer (ShortConv) without a
    quant_config, so its in_proj/out_proj are unquantized at serving time.
    The profile must pin them (BF16 passthrough) or the exported artifact
    fails to load (KeyError on …short_conv.out_proj.input_global_scale)."""
    pins = list(Lfm2MoeProfile().pinned_names())
    assert "conv.in_proj" in pins
    assert "conv.out_proj" in pins
    # depthwise conv + router/bias stay pinned too
    assert "conv.conv" in pins


def test_missing_expert_fails_loud():
    """A gap in the expert index range must raise."""
    prof, pat = _lfm_pat()
    I, H = 6, 8
    blk = "model.layers.2.feed_forward.experts"
    # experts 0 and 2 present, 1 missing
    out = {}
    for i in (0, 2):
        out[f"{blk}.{i}.w1.weight"] = torch.randn(I, H)
        out[f"{blk}.{i}.w3.weight"] = torch.randn(I, H)
        out[f"{blk}.{i}.w2.weight"] = torch.randn(H, I)
    live = {f"{blk}.gate_up_proj": (3, 2 * I, H), f"{blk}.down_proj": (3, H, I)}
    try:
        _pack(out, prof, pat, live)
    except ValueError as e:
        assert "missing expert" in str(e)
    else:
        raise AssertionError("expected ValueError on missing expert")
