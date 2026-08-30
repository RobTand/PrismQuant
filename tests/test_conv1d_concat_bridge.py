"""Generic N->1 concat source bridge (`_merge_concat_sources`).

Some checkpoints store one live parameter as several separate tensors that the
modelling code concatenates on load — transformers' ``Concatenate(dim=...)``
merges. glm5_next is the motivating case: each of its 34 KDA layers ships
``self_attn.{q,k,v}_conv1d.weight`` while the live module holds one fused
``self_attn.conv1d.weight``. ``checkpoint_to_live_name`` is a 1:1-or-drop
contract and cannot express that, so the merge is declared in the profile spec
(``concat_merges``) and executed by the loader bridge — no architecture names
in ``layer_streaming``.

These tests pin the exact layout (order, shape, dtype) and the safety
behaviours, on synthetic tensors only: nothing here reads the GLM checkpoint.
"""
import pytest
import torch

from prismaquant.layer_streaming import _merge_concat_sources
from prismaquant.model_profiles.glm5_next import Glm5NextProfile

# The KDA short convolution, shrunk. Real shape is [8192, 1, 4] per source.
CONV_DIM, KERNEL = 6, 4
LAYER = "model.language_model.layers.0.self_attn"


def _glm_groups():
    """The declaration under test, read from the shipped profile/spec — not
    re-typed here, so a spec edit that breaks the contract fails this test."""
    return Glm5NextProfile().concat_merge_groups()


def _sources(dtype=torch.bfloat16, fill=(1.0, 2.0, 3.0)):
    """q/k/v filled with distinguishable constants, so a wrong concat order
    is visible in the merged tensor rather than hidden by random data."""
    return {
        f"{LAYER}.{leaf}_conv1d.weight": torch.full(
            (CONV_DIM, 1, KERNEL), value, dtype=dtype)
        for leaf, value in zip("qkv", fill)
    }


def _live(shape=(3 * CONV_DIM, 1, KERNEL)):
    return {f"{LAYER}.conv1d.weight": shape}


# --------------------------------------------------------------- declaration


def test_profile_declares_the_qkv_conv1d_merge():
    """The order is transformers' (`conversion_mapping.py` "glm5_next":
    `Concatenate(dim=0)` over q, k, v) and is load-bearing."""
    assert _glm_groups() == (
        (
            "self_attn.conv1d.weight",
            (
                "self_attn.q_conv1d.weight",
                "self_attn.k_conv1d.weight",
                "self_attn.v_conv1d.weight",
            ),
            0,
        ),
    )


def test_profile_reports_no_unbridged_source_keys():
    """The conv1d gap was this profile's only one; it is now declared."""
    assert Glm5NextProfile().unbridged_source_keys() == ()


# --------------------------------------------------------------- the merge


def test_merge_equals_torch_cat_in_declared_order():
    out = _sources()
    expected = torch.cat(
        [out[f"{LAYER}.{leaf}_conv1d.weight"] for leaf in "qkv"], dim=0)

    n = _merge_concat_sources(
        out, groups=_glm_groups(), live_param_shape=_live().get)

    assert n == 1
    merged = out[f"{LAYER}.conv1d.weight"]
    assert torch.equal(merged, expected)
    # Order-sensitive: the three blocks appear q, then k, then v.
    assert torch.equal(merged[:CONV_DIM], torch.full_like(merged[:CONV_DIM], 1.0))
    assert torch.equal(merged[CONV_DIM:2 * CONV_DIM],
                       torch.full_like(merged[:CONV_DIM], 2.0))
    assert torch.equal(merged[2 * CONV_DIM:],
                       torch.full_like(merged[:CONV_DIM], 3.0))
    # A reversed cat would NOT satisfy the above — pin that explicitly.
    assert not torch.equal(
        merged,
        torch.cat([out_ for out_ in reversed(
            [torch.full((CONV_DIM, 1, KERNEL), v, dtype=torch.bfloat16)
             for v in (1.0, 2.0, 3.0)])], dim=0))


def test_merge_preserves_dtype_and_shape():
    """The conv1d weights are BF16 in the GLM checkpoint; the bridge must not
    cast or promote."""
    out = _sources(dtype=torch.bfloat16)
    _merge_concat_sources(
        out, groups=_glm_groups(), live_param_shape=_live().get)
    merged = out[f"{LAYER}.conv1d.weight"]
    assert merged.dtype == torch.bfloat16
    assert tuple(merged.shape) == (3 * CONV_DIM, 1, KERNEL)
    assert merged.is_contiguous()


def test_sources_are_consumed_and_bystanders_untouched():
    out = _sources()
    out[f"{LAYER}.q_proj.weight"] = torch.zeros(2, 2)
    _merge_concat_sources(
        out, groups=_glm_groups(), live_param_shape=_live().get)
    assert set(out) == {f"{LAYER}.conv1d.weight", f"{LAYER}.q_proj.weight"}


def test_merges_every_layer_independently():
    groups = _glm_groups()
    out, live = {}, {}
    for layer in (0, 7, 12):
        base = f"model.language_model.layers.{layer}.self_attn"
        for leaf, value in zip("qkv", (1.0, 2.0, 3.0)):
            out[f"{base}.{leaf}_conv1d.weight"] = torch.full(
                (CONV_DIM, 1, KERNEL), value, dtype=torch.bfloat16)
        live[f"{base}.conv1d.weight"] = (3 * CONV_DIM, 1, KERNEL)

    assert _merge_concat_sources(
        out, groups=groups, live_param_shape=live.get) == 3
    assert set(out) == set(live)


# --------------------------------------------------------------- safety


def test_no_live_target_leaves_sources_alone():
    """A live module that already holds the split layout has no gap to
    bridge; the bridge must not invent a parameter."""
    out = _sources()
    before = dict(out)
    assert _merge_concat_sources(
        out, groups=_glm_groups(), live_param_shape={}.get) == 0
    assert out == before


def test_missing_source_raises():
    out = _sources()
    del out[f"{LAYER}.k_conv1d.weight"]
    with pytest.raises(ValueError, match="missing source tensor"):
        _merge_concat_sources(
            out, groups=_glm_groups(), live_param_shape=_live().get)


def test_mixed_source_dtypes_raise_instead_of_promoting():
    out = _sources()
    out[f"{LAYER}.v_conv1d.weight"] = out[
        f"{LAYER}.v_conv1d.weight"].to(torch.float32)
    with pytest.raises(ValueError, match="mixed dtypes"):
        _merge_concat_sources(
            out, groups=_glm_groups(), live_param_shape=_live().get)


def test_wrong_assembled_shape_raises():
    """The live-parameter shape check is the safety net against a bad
    declaration (wrong dim, wrong source set)."""
    out = _sources()
    wrong = _live(shape=(2 * CONV_DIM, 1, KERNEL))
    with pytest.raises(ValueError, match="!= live param"):
        _merge_concat_sources(
            out, groups=_glm_groups(), live_param_shape=wrong.get)


def test_non_zero_dim_is_honored():
    """`dim` comes from the declaration, not from the shapes."""
    groups = (("blk.fused", ("blk.a", "blk.b"), 1),)
    out = {"blk.a": torch.ones(2, 3), "blk.b": torch.zeros(2, 5)}
    assert _merge_concat_sources(
        out, groups=groups, live_param_shape={"blk.fused": (2, 8)}.get) == 1
    assert tuple(out["blk.fused"].shape) == (2, 8)


def test_no_declaration_is_a_no_op():
    out = _sources()
    before = dict(out)
    assert _merge_concat_sources(
        out, groups=(), live_param_shape=_live().get) == 0
    assert out == before


def test_nested_source_suffixes_attribute_to_the_longer_one():
    """A declaration whose suffixes nest (`a.w` is a tail of `b.a.w`) must not
    depend on declaration order."""
    groups = (("blk.fused", ("blk.b.a.w", "blk.a.w"), 0),)
    out = {"blk.b.a.w": torch.ones(2, 3), "blk.a.w": torch.zeros(2, 3)}
    assert _merge_concat_sources(
        out, groups=groups, live_param_shape={"blk.fused": (4, 3)}.get) == 1
    merged = out["blk.fused"]
    assert torch.equal(merged[:2], torch.ones(2, 3))
    assert torch.equal(merged[2:], torch.zeros(2, 3))
