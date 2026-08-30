"""One dense family, two serving classes, two namespaces.

Qwen3.5/3.6 dense ships as a multimodal wrapper
(`Qwen3_5ForConditionalGeneration`) and as a text-only carve-out
(`Qwen3_5ForCausalLM`).  vLLM builds them under DIFFERENT module names, and the
difference is not cosmetic: it decides what a `config_groups` target must say.

    wrapper    lm_head -> language_model.lm_head   model.X -> language_model.model.X
    text-only  lm_head -> lm_head                  model.X -> model.X

`Qwen3_5ForCausalLMBase.hf_to_vllm_mapper` is `{"model.language_model.":
"model."}` — it STRIPS the wrapper prefix rather than adding it — and it does
not touch `lm_head.` at all, so the head is a bare `lm_head`.

This is not a hypothetical.  Artifact A (Qwen3.8-27B, 12.98 GB, text-only) was
exported with the wrapper's namespace, so its one delegated compressed-tensors
group targeted `re:^language_model[.]lm_head$`, matched no module, left the
head unquantized, and died at load with an orphaned `lm_head.weight_scale`.
One wrong string out of a 12.98 GB artifact, discoverable only by serving it.

The regression bar is both directions: a text-only checkpoint must get the
text-only names, and a multimodal checkpoint must keep the names every already
-shipped Qwen3.5/3.6 artifact was built with.
"""
from __future__ import annotations

import pytest

from prismaquant.model_profiles.qwen3_5_dense import Qwen3_5DenseProfile
from prismaquant.model_profiles.registry import profile_from_config

TEXT_ONLY = {"model_type": "qwen3_5_text",
             "architectures": ["Qwen3_5ForCausalLM"]}
MULTIMODAL = {"model_type": "qwen3_5",
              "architectures": ["Qwen3_5ForConditionalGeneration"]}


def test_text_only_checkpoint_gets_the_text_only_namespace():
    p = profile_from_config(TEXT_ONLY)
    assert isinstance(p, Qwen3_5DenseProfile)
    assert p.to_vllm_internal_name("lm_head") == "lm_head"
    assert p.to_vllm_internal_name("model.embed_tokens") == "model.embed_tokens"
    assert (p.to_vllm_internal_name("model.layers.0.self_attn.q_proj")
            == "model.layers.0.self_attn.q_proj")
    # The checkpoint spelling is also accepted and normalized down, because the
    # exporter may hold either the recipe or the on-disk name.
    assert (p.to_vllm_internal_name("model.language_model.layers.0.mlp.down_proj")
            == "model.layers.0.mlp.down_proj")


def test_multimodal_checkpoint_keeps_the_wrapper_namespace():
    p = profile_from_config(MULTIMODAL)
    assert p.to_vllm_internal_name("lm_head") == "language_model.lm_head"
    assert (p.to_vllm_internal_name("model.embed_tokens")
            == "language_model.model.embed_tokens")
    assert (p.to_vllm_internal_name("model.layers.0.self_attn.q_proj")
            == "language_model.model.layers.0.self_attn.q_proj")
    assert (p.to_vllm_internal_name("model.visual.blocks.0.attn.qkv")
            == "visual.blocks.0.attn.qkv")


def test_an_undeclared_profile_keeps_the_historical_answer():
    """A hand-built profile declares nothing, and must not change behavior.

    Every caller that constructs the profile directly (tests, older entry
    points) predates the declaration, so the wrapper namespace stays the
    default.  This is what makes the fix non-regressive for artifacts already
    on disk.
    """
    p = Qwen3_5DenseProfile()
    assert p.declared_architectures() == ()
    assert p.vllm_architecture_class() == "Qwen3_5ForConditionalGeneration"
    assert p.to_vllm_internal_name("lm_head") == "language_model.lm_head"


@pytest.mark.parametrize("cfg,expected", [
    (TEXT_ONLY, "Qwen3_5ForCausalLM"),
    (MULTIMODAL, "Qwen3_5ForConditionalGeneration"),
    ({"model_type": "qwen3_6", "architectures": ["Qwen3_6ForCausalLM"]},
     "Qwen3_5ForCausalLM"),
])
def test_vllm_class_follows_the_declaration(cfg, expected):
    """The class the profile reads metadata from must be the class that serves.

    `to_vllm_internal_name` consults the spec first and the vLLM class's
    `hf_to_vllm_mapper` second.  If only the spec were fixed, the derivation
    would be correct on a host without vLLM and wrong on a host with it —
    a build box and a serving box would disagree about the artifact.
    """
    assert profile_from_config(cfg).vllm_architecture_class() == expected


def test_the_checkpoint_namespace_does_not_move():
    """`source_tensor_name` is spec-structural, not class-derived.

    The tensors legitimately live at `model.language_model.*` on disk for BOTH
    classes — that is exactly what the text-only class's mapper exists to strip.
    A fix that moved them would have rewritten 1365 tensor names and broken
    byte-identity with the artifact already built.
    """
    text = profile_from_config(TEXT_ONLY)
    multi = profile_from_config(MULTIMODAL)
    for name in ("lm_head", "model.embed_tokens",
                 "model.layers.0.self_attn.q_proj"):
        assert text.source_tensor_name(name) == multi.source_tensor_name(name)
    assert (text.source_tensor_name("model.embed_tokens")
            == "model.language_model.embed_tokens")


# Recorded FROM vLLM, not from the spec, inside the serving image
# `gridbook:0.8.7-clean-98916b0` (vllm 0.26.1rc1.dev693+g7f7a32cfe.d20260812) on
# 2026-08-15 by reading `cls.hf_to_vllm_mapper.orig_to_new_prefix` for each
# class.  The lane has since moved to `gridbook:0.8.8-clean-064a4cb`, which is
# built FROM that image and carries the same vLLM build, so the fixture still
# describes the runtime that serves.  This is the only fixture here that is independent of PrismaQuant's own
# map: every other check in this file, and every static check that passed before
# the failed serve, takes the producer's spelling as its input and therefore
# cannot falsify it.  If vLLM changes these prefixes, this fixture is what
# should fail.
VLLM_PREFIX_MAP = {
    "Qwen3_5ForCausalLM": {"model.language_model.": "model."},
    "Qwen3_5ForConditionalGeneration": {
        "model.visual.": "visual.",
        "lm_head.": "language_model.lm_head.",
        "model.language_model.": "language_model.model.",
    },
}


def _apply_vllm_prefixes(name: str, prefix_map: dict[str, str]) -> str:
    """vLLM's own rewrite: longest matching prefix wins, `lm_head` is only
    touched if a rule names it, and an unmatched name passes through."""
    for src in sorted(prefix_map, key=len, reverse=True):
        if name.startswith(src):
            return prefix_map[src] + name[len(src):]
        # vLLM's mapper keys are dotted prefixes; a bare leaf like `lm_head`
        # is matched by the `lm_head.` rule only once a suffix is appended,
        # which is how the head's PARAMETERS are remapped.
        if src.endswith(".") and name + "." == src:
            return prefix_map[src].rstrip(".")
    return name


@pytest.mark.parametrize("cfg,arch", [(TEXT_ONLY, "Qwen3_5ForCausalLM"),
                                      (MULTIMODAL, "Qwen3_5ForConditionalGeneration")])
def test_our_map_agrees_with_the_serving_runtimes_own_map(cfg, arch):
    """The spec must say what vLLM will actually do — checked against vLLM."""
    profile = profile_from_config(cfg)
    prefix_map = VLLM_PREFIX_MAP[arch]
    for name in ("lm_head", "model.embed_tokens",
                 "model.language_model.layers.0.self_attn.q_proj"):
        # Compare on the CHECKPOINT spelling, which is what vLLM's loader sees.
        checkpoint = profile.source_tensor_name(name)
        assert (profile.to_vllm_internal_name(checkpoint)
                == _apply_vllm_prefixes(checkpoint, prefix_map)), name


def test_a_naming_variant_must_declare_a_condition_and_a_map():
    """The variant vocabulary refuses the two shapes that would silently no-op."""
    from prismaquant.model_profiles.structure import NamingVariant

    with pytest.raises(ValueError, match="must declare"):
        NamingVariant.from_dict({"naming": {"recipe_to_vllm": []}})
    with pytest.raises(ValueError, match="at least one map"):
        NamingVariant.from_dict({"when": {"architectures": ["X"]}})
    with pytest.raises(ValueError, match="unsupported"):
        NamingVariant.from_dict({
            "when": {"architectures": ["X"]}, "nameing": {}})


def test_qwen3_5_dense_and_moe_are_the_only_specs_with_naming_variants():
    """Every OTHER spec is untouched by the mechanism — checked over all of them.

    The DSv4 W8A16 export handoff's frozen-source review cites this exact
    property to justify re-freezing `base.py`/`registry.py`: if no other spec
    declares a variant, `for_config` is never reached on any other lane and the
    spec those profiles return is the file spec. A review note is only worth
    what its assertion covers, so this enumerates the directory rather than a
    hand-picked three.
    """
    import json
    import pathlib

    from prismaquant.model_profiles import structure as structure_mod

    specs_dir = pathlib.Path(structure_mod.__file__).parent / "specs"
    files = sorted(specs_dir.glob("*.json"))
    assert files, f"no structure specs found under {specs_dir}"
    with_variants = sorted(
        p.name for p in files
        if json.loads(p.read_text()).get("naming_variants")
    )
    assert with_variants == [
        "qwen3_5.json",
        "qwen3_5_dense.json",
    ], with_variants


def test_specs_without_variants_are_returned_unchanged():
    """`for_config` is a no-op — identically `self` — without variants."""
    from prismaquant.model_profiles.structure import load_structure_spec

    for spec_id in ("gemma4", "minimax_m2", "deepseek_v4"):
        spec = load_structure_spec(spec_id)
        if spec is None:
            continue
        assert spec.naming_variants == ()
        assert spec.for_config("anything", ["Anything"]) is spec
