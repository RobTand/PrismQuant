"""glm5_next streamed forward must reproduce the upstream forward exactly.

GLM-5.3-Flash is the first architecture in this tree that runs its decoder
stack on `hc_mult` parallel residual streams (manifold-constrained
hyper-connections) AND declares a `deepseek_sparse_attention` layer type whose
indexer consumes a 2D boolean padding mask rather than a dense additive causal
mask.  Both are wired for the streamed pass in exactly two places:

* `Glm5NextProfile.expand_hidden_for_layers` / `collapse_hidden_after_layers`
  (`prismaquant/model_profiles/glm5_next.py`), and
* the `deepseek_sparse_attention` branch of
  `prismaquant.layer_streaming._compute_attention_mask`.

Neither is checkable by inspection: a wrong mask is a plausible-looking tensor
and a wrong stream collapse is a plausible-looking mean.  The only honest test
is logit parity against the reference implementation on the same weights, which
is what this module does -- on a tiny random config, on CPU, in seconds, so the
wiring is verified before a GPU forward-fidelity gate is ever launched.

The streamed side drives the real `StreamedCausalLM` over a stub context: the
class reads only module handles and residency callbacks from its context, and
substituting resident modules for streamed ones is precisely the identity under
test (a genuine `StreamingContext` would add safetensors residency, not forward
semantics).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

glm5 = pytest.importorskip(
    "transformers.models.glm5_next",
    reason="glm5_next requires transformers >= 5.16",
)

from prismaquant import genuine_weight_initialization  # noqa: E402
from prismaquant.cost_streaming import StreamedCausalLM  # noqa: E402
from prismaquant.model_profiles.glm5_next import Glm5NextProfile  # noqa: E402


def _tiny_config():
    """A two-layer glm5_next: one KDA layer, one DSA/MLA layer."""
    text = glm5.Glm5NextTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=0,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=0,
        v_head_dim=16,
        mla_use_nope=True,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
        index_head_dim=16,
        index_n_heads=2,
        index_topk=8,
        index_kpool=2,
        index_kpool_compress=True,
        linear_attn_config={
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        mhc=True,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        hc_eps=1e-6,
        num_nextn_predict_layers=0,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
        dtype=torch.float32,
    )
    vision = glm5.Glm5NextVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        projection_intermediate_size=64,
        out_hidden_size=64,
        depth=1,
        num_heads=2,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    return glm5.Glm5NextConfig(
        text_config=text,
        vision_config=vision,
        dtype=torch.float32,
    )


def _build_model(config):
    """Construct the reference model with every parameter initialized.

    `genuine_weight_initialization` is load-bearing, not decoration.
    Importing anything from `prismaquant` no-ops
    `PreTrainedModel._initialize_weights` process-wide
    (`prismaquant/__init__.py::_polyfill_transformers`), which is sound for
    PrismaQuant's own loaders -- they overwrite every parameter from a
    checkpoint straight afterwards -- and unsound here, where the
    from-config model IS the subject.  Without the restore, every tensor
    the modeling file allocates as a bare `nn.Parameter(torch.empty(...))`
    keeps whatever the allocator last left in that page: on this config the
    routed-expert `mlp.experts.gate_up_proj` / `down_proj`, measured at
    ~2e17 on one run and non-finite on another.  Those weights feed BOTH
    sides of the parity comparison, so a bad page NaNs the reference and
    the streamed pass alike and the divergence reads `nan` at every context
    octave -- which is how it reached CI (run 33284249771, py3.11 failed
    and py3.12 passed on the same commit, same torch and transformers).

    A parity assertion cannot see that: `nan != nan` on both sides looks
    exactly like a wiring defect.  So the finiteness check below fails
    closed and names the mechanism instead.
    """
    with genuine_weight_initialization():
        model = glm5.Glm5NextForConditionalGeneration(config)
    model = model.to(torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # `_init_weights` leaves several mHC / KDA parameters at a constant;
    # a degenerate (all-zero or all-equal) tensor would make a wrong wiring
    # agree with a right one by accident.
    for name, parameter in model.named_parameters():
        if name.endswith((".fn", ".base", ".scale", ".dt_bias", ".A_log")):
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    unset = [
        name
        for name, tensor in (
            list(model.named_parameters()) + list(model.named_buffers())
        )
        if tensor is not None and not torch.isfinite(tensor).all()
    ]
    assert not unset, (
        "the reference model was built with non-finite tensors, so both "
        "sides of the parity comparison are garbage and the divergence "
        "below would read `nan` rather than localize a wiring defect; "
        f"offending tensors: {unset[:8]}"
    )
    return model


def _build_tiny_model():
    torch.manual_seed(20260826)
    return _build_model(_tiny_config())


def _streamed_runner(model):
    base_model = model.model.language_model
    context = SimpleNamespace(
        model=model,
        base_model=base_model,
        layers=base_model.layers,
        layers_prefix="model.language_model.layers.",
        num_layers=len(base_model.layers),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cache_slots=1,
        install=lambda layer, **_kwargs: None,
        unload=lambda layer: None,
        schedule_prefetch=lambda layer: None,
    )
    return StreamedCausalLM(
        context,
        Glm5NextProfile(),
        prefetch_lookahead=0,
        require_prefetched_residency=False,
    )


@pytest.fixture(autouse=True)
def _torch_only_causal_conv1d(monkeypatch):
    """Force the pure-torch short convolution so this runs on CPU.

    `causal_conv1d` and `fla` are installed in the build venv, and
    transformers binds their CUDA entry points at import time; on CPU they
    raise rather than falling back.  Both sides of the parity comparison
    call the same module-level names, so substituting the reference
    implementations transformers itself ships (`__wrapped__`, the
    undecorated functions) changes the kernel for both and leaves the
    identity under test -- the streamed wiring -- exactly as it is.
    """
    from transformers.models.glm5_next import modeling_glm5_next as upstream

    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_kimi_delta_attention",
        "recurrent_kimi_delta_attention",
    ):
        bound = getattr(upstream, name)
        torch_only = getattr(bound, "__wrapped__", None)
        if torch_only is None:
            pytest.skip(f"cannot recover the torch-only {name}")
        monkeypatch.setattr(upstream, name, torch_only)


@pytest.fixture(scope="module")
def tiny_model():
    return _build_tiny_model()


@pytest.mark.parametrize("shape", [(1, 12), (3, 9)])
def test_streamed_logits_match_upstream_forward(tiny_model, shape):
    """The streamed pass and `Glm5NextForConditionalGeneration` agree."""
    torch.manual_seed(11)
    ids = torch.randint(0, tiny_model.config.text_config.vocab_size, shape)

    with torch.inference_mode():
        reference = tiny_model(input_ids=ids).logits
        streamed = _streamed_runner(tiny_model)(ids).logits

    assert streamed.shape == reference.shape
    assert torch.isfinite(streamed).all()
    torch.testing.assert_close(streamed, reference, rtol=1e-4, atol=1e-4)


def test_dsa_layers_receive_a_boolean_padding_mask(tiny_model):
    """The DSA entry is a 2D bool `[B, S]` mask, never a dense causal one."""
    from prismaquant.layer_streaming import _compute_attention_mask

    base_model = tiny_model.model.language_model
    hidden = torch.zeros(2, 7, base_model.config.hidden_size)
    position_ids = torch.arange(7).unsqueeze(0)
    masks = _compute_attention_mask(base_model, hidden, position_ids)

    assert isinstance(masks, dict)
    for layer_type in base_model.config.layer_types:
        assert layer_type in masks, (
            f"{layer_type!r} is absent from the streamed mask mapping; "
            "_call_layer would fail closed on it"
        )
    dsa = masks["deepseek_sparse_attention"]
    assert dsa is not None
    assert dsa.dtype is torch.bool
    assert tuple(dsa.shape) == (2, 7)
    assert bool(dsa.all())


def test_stream_expansion_round_trips_through_the_profile(tiny_model):
    """expand/collapse mirror `hc_mult` expansion and `hc_head`."""
    profile = Glm5NextProfile()
    base_model = tiny_model.model.language_model
    hidden = torch.randn(2, 5, base_model.config.hidden_size)

    expanded = profile.expand_hidden_for_layers(hidden, base_model)
    assert tuple(expanded.shape) == (
        2, 5, base_model.config.hc_mult, base_model.config.hidden_size,
    )
    collapsed = profile.collapse_hidden_after_layers(expanded, base_model)
    torch.testing.assert_close(collapsed, hidden)


def test_stream_hooks_refuse_a_wrong_rank(tiny_model):
    profile = Glm5NextProfile()
    base_model = tiny_model.model.language_model
    with pytest.raises(RuntimeError, match="rank-3"):
        profile.expand_hidden_for_layers(
            torch.randn(2, 5, 4, base_model.config.hidden_size), base_model
        )
    with pytest.raises(RuntimeError, match="rank-4"):
        profile.collapse_hidden_after_layers(
            torch.randn(2, 5, base_model.config.hidden_size), base_model
        )


# ----------------------------------------------------------------------
# Long-sequence parity -- the pinpointing tool
# ----------------------------------------------------------------------
#
# The 12-token cases above cannot see a defect that only appears with depth,
# and depth is exactly where this architecture is fragile: the KDA linear
# attention runs a CHUNKED scan whose recurrent state carries across chunk
# boundaries, the short convolution carries a 4-wide window across the same
# boundaries, and the DSA indexer only starts actually sparsifying once the
# sequence exceeds `index_topk`.  At 12 tokens none of those three is
# exercised; a streamed pass that drops or resets state between chunks is
# bit-identical to a correct one.
#
# So the long case runs past all three thresholds and reports WHERE the two
# implementations start to disagree, by context octave.  A wiring defect
# shows up as divergence that switches on at a specific position and grows;
# floating-point noise is flat and tiny.  Isolating the layer types tells
# you which of the three suspects owns it.

_LONG_SEQLEN = 1024


def _divergence_by_octave(streamed, reference):
    """Max absolute logit divergence per context octave."""
    difference = (streamed - reference).abs().amax(dim=-1).amax(dim=0)
    profile, first, length = [], 0, 1
    while first < difference.numel():
        last = min(first + length, difference.numel())
        profile.append({
            "context_first": first + 1,
            "context_last": last,
            "max_abs_divergence": float(difference[first:last].max().item()),
        })
        first, length = last, length * 2
    return profile


def _format_divergence(profile):
    lines = ["  context        max|dlogit|"]
    for block in profile:
        lines.append(
            f"  {block['context_first']:>5d}-{block['context_last']:<5d} "
            f"{block['max_abs_divergence']:>15.3e}"
        )
    return "\n".join(lines)


def _long_parity(model, seqlen=_LONG_SEQLEN):
    torch.manual_seed(4242)
    ids = torch.randint(0, model.config.text_config.vocab_size, (1, seqlen))
    with torch.inference_mode():
        # `use_cache=False` matters for the isolated variants below: with
        # every layer a linear-attention layer, upstream's DynamicCache
        # raises StopIteration from `get_seq_length`, which searches for an
        # attention layer to read a length from and finds none.  That is a
        # limitation of the reference cache on a degenerate config -- the
        # real model always carries 11 DSA layers -- and not something the
        # streamed pass is party to, so the comparison is run cacheless on
        # both sides rather than skipped.  The streamed pass never builds a
        # cache, so this also makes the two sides more alike, not less.
        reference = model(input_ids=ids, use_cache=False).logits
        streamed = _streamed_runner(model)(ids).logits
    return streamed, reference, _divergence_by_octave(streamed, reference)


def test_streamed_logits_match_upstream_at_long_sequence(tiny_model):
    """Parity must hold past the chunk, conv-window and top-k thresholds."""
    config = tiny_model.config.text_config
    assert _LONG_SEQLEN > config.index_topk, (
        "the long case must exceed index_topk or the DSA indexer never "
        "sparsifies and the test proves nothing about long range"
    )

    streamed, reference, profile = _long_parity(tiny_model)

    assert streamed.shape == reference.shape
    assert torch.isfinite(streamed).all()
    worst = max(block["max_abs_divergence"] for block in profile)
    assert worst < 1e-3, (
        f"streamed/upstream logits diverge by {worst:.3e} at "
        f"{_LONG_SEQLEN} tokens; divergence by context octave:\n"
        + _format_divergence(profile)
    )


@pytest.mark.parametrize(
    "layer_types,mlp_layer_types",
    [
        (["linear_attention", "linear_attention"], ["dense", "sparse"]),
        (
            ["deepseek_sparse_attention", "deepseek_sparse_attention"],
            ["dense", "sparse"],
        ),
    ],
    ids=["kda_only", "dsa_only"],
)
def test_long_sequence_parity_per_layer_type(layer_types, mlp_layer_types):
    """Same long case with one layer type at a time, to localize a defect.

    If the mixed model above ever fails, whichever of these two also fails
    names the suspect: the KDA chunked/recurrent state and its short
    convolution, or the DSA mask semantics at long range.
    """
    torch.manual_seed(20260826)
    config = _tiny_config()
    config.text_config.layer_types = list(layer_types)
    config.text_config.mlp_layer_types = list(mlp_layer_types)
    config.text_config.indexer_types = ["full", "full"]
    model = _build_model(config)

    streamed, reference, profile = _long_parity(model)
    worst = max(block["max_abs_divergence"] for block in profile)
    assert worst < 1e-3, (
        f"{layer_types[0]} diverges by {worst:.3e} at {_LONG_SEQLEN} "
        f"tokens; divergence by context octave:\n" + _format_divergence(profile)
    )
