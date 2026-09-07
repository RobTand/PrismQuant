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
    # Mirror the production predicate (_build_expert_packer): match the raw
    # checkpoint key or its remap to vLLM scheme-dispatch naming, so both
    # checkpoint-named and multimodal specs are handled.
    def is_per_expert(name):
        return bool(pat.match(name)) or bool(pat.match(prof.to_vllm_internal_name(name)))
    return _pack_per_expert_into_packed(
        out,
        is_per_expert=is_per_expert,
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


def test_multimodal_naming_qwen35():
    """Regression: multimodal specs author `per_expert_regex` in vLLM
    scheme-dispatch naming (`language_model.model.layers.*`) while the
    checkpoint/`out` keys are in HF naming (`model.language_model.layers.*`).
    The bridge must still detect + pack per-expert tensors via the profile's
    name remap. This is the Ornith-1.0-35B (Qwen3.5-MoE, per-expert on disk)
    case; Qwen3.6-35B-A3B shipped packed so it never exercised this path."""
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
    prof = Qwen3_5Profile()
    regex = prof.per_expert_moe_regex()
    pat = re.compile(regex[len("re:"):] if regex.startswith("re:") else regex)
    E, I, H = 3, 4, 8  # experts, moe_intermediate, hidden
    blk = "model.language_model.layers.0.mlp.experts"  # HF checkpoint naming
    out, gates, ups, downs = {}, {}, {}, {}
    for i in range(E):
        g, u, d = torch.randn(I, H), torch.randn(I, H), torch.randn(H, I)
        out[f"{blk}.{i}.gate_proj.weight"] = g
        out[f"{blk}.{i}.up_proj.weight"] = u
        out[f"{blk}.{i}.down_proj.weight"] = d
        gates[i], ups[i], downs[i] = g, u, d
    live = {f"{blk}.gate_up_proj": (E, 2 * I, H), f"{blk}.down_proj": (E, H, I)}
    n = _pack(out, prof, pat, live)
    assert n == 2, f"expected 2 packed params, got {n}"
    # per-expert keys consumed, packed keys inserted
    assert not any(".experts.0." in k for k in out)
    gu = out[f"{blk}.gate_up_proj"]
    dp = out[f"{blk}.down_proj"]
    assert tuple(gu.shape) == (E, 2 * I, H)
    assert tuple(dp.shape) == (E, H, I)
    # gate on top, up on bottom, experts stacked in index order
    for i in range(E):
        assert torch.allclose(gu[i, :I], gates[i])
        assert torch.allclose(gu[i, I:], ups[i])
        assert torch.allclose(dp[i], downs[i])


def test_pack_releases_consumed_sources_and_refuses_dtype_promotion():
    import weakref
    import pytest
    prof, pat = _lfm_pat()
    blk = 'model.layers.2.feed_forward.experts'
    live = {f'{blk}.gate_up_proj':(2,12,8),f'{blk}.down_proj':(2,8,6)}
    out = {f'{blk}.{i}.{proj}.weight':torch.randn(shape,dtype=torch.bfloat16)
           for i in range(2) for proj,shape in [('w1',(6,8)),('w3',(6,8)),('w2',(8,6))]}
    refs = [weakref.ref(t) for t in out.values()]
    assert _pack(out,prof,pat,live)==2
    assert all(ref() is None for ref in refs)
    assert all(t.dtype==torch.bfloat16 for t in out.values())
    bad = {f'{blk}.0.w1.weight':torch.randn(6,8,dtype=torch.bfloat16),
           f'{blk}.0.w3.weight':torch.randn(6,8,dtype=torch.float32)}
    original = dict(bad)
    with pytest.raises(ValueError,match='dtype/device'):
        _pack(bad,prof,pat,{f'{blk}.gate_up_proj':(1,12,8)})
    assert bad.keys()==original.keys()
    assert all(bad[name] is value for name,value in original.items())
