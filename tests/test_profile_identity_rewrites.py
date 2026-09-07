"""Declared identity rewrites must remain authoritative at scheme dispatch."""
from dataclasses import replace

import pytest

from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.model_profiles.qwen3_5_dense import Qwen3_5DenseProfile
from prismaquant.model_profiles.registry import profile_from_config
from prismaquant.model_profiles.structure import NameRewriteRule
from prismaquant.model_profiles.vllm_registry import name_remapper_from_prefix_map


@pytest.mark.parametrize("profile_type", [Qwen3_5Profile, Qwen3_5DenseProfile])
@pytest.mark.parametrize("runtime_destination", [None, "model."])
def test_declared_mtp_identity_survives_runtime_loader_drop_or_rename(profile_type, runtime_destination):
    profile = profile_type()
    profile._name_remapper = name_remapper_from_prefix_map({"mtp.": runtime_destination})
    name = "mtp.layers.0.mlp.experts.down_proj"
    assert profile.to_vllm_internal_name(name) == name


def test_native_body_identity_is_not_overwritten_by_runtime_fallback():
    profile = profile_from_config({"model_type": "qwen3_5_moe_text",
                                   "architectures": ["Qwen3_5MoeForCausalLM"]})
    profile._name_remapper = name_remapper_from_prefix_map({"model.": "language_model.model."})
    name = "model.layers.0.mlp.experts.down_proj"
    assert profile.to_vllm_internal_name(name) == name


def test_unmatched_name_still_uses_runtime_mapper():
    profile = Qwen3_5Profile()
    profile._name_remapper = name_remapper_from_prefix_map({"other.": "runtime.other."})
    assert profile.to_vllm_internal_name("other.proj") == "runtime.other.proj"


def test_matched_rewrite_chain_returning_original_name_remains_authoritative(monkeypatch):
    profile = Qwen3_5Profile()
    spec = replace(profile.structure_spec(), recipe_to_vllm=(
        NameRewriteRule(prefix="model.", replace="temporary."),
        NameRewriteRule(prefix="temporary.", replace="model.", stop=True),
    ))
    monkeypatch.setattr(profile, "structure_spec", lambda: spec)
    profile._name_remapper = name_remapper_from_prefix_map({"model.": "wrong."})
    assert profile.to_vllm_internal_name("model.proj") == "model.proj"
