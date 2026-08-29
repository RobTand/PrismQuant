"""R8 — the strict gate for `spec.match` and `SpecMatchProfile`.

Detection order governs which profile a shipped checkpoint resolves to;
getting it wrong silently re-routes every future 27B/35B run to another
profile (the fused-coherence class of bug that shipped unservable artifacts).
So `SpecMatchProfile` lands *alongside* the Python profiles, not instead of
them, and this file is the mitigation: for every registered profile, on every
representative config in the family, the **spec verdict must equal the Python
verdict**. Only once that is green for a release does a `matches()` body get
deleted, one architecture at a time.

The equivalence domain is single-architecture configs — every real
`config.json` in these families lists exactly one entry. Outside it the two
forms can legitimately differ: `qwen3_5_dense.py` scans the list and returns
on the *first* interesting entry, so a hypothetical mixed
`["Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"]` would claim the dense
profile, while `architectures_exclude` vetoes it. The declarative answer is
the safer one, and the ambiguity is not reachable from any shipped
checkpoint.
"""
from __future__ import annotations

import pytest

from prismaquant.model_profiles import registry as _registry
from prismaquant.model_profiles.spec_profile import SpecMatchProfile
from prismaquant.model_profiles.structure import (
    SPEC_DEFAULT_PRIORITY,
    SpecMatch,
    iter_structure_specs,
    load_structure_spec,
)

PROFILE_CLASSES = list(_registry._REGISTERED)

# Today's literal `_REGISTERED` order. `priority` replaced the two ordering
# comments that used to encode it; this list is what those comments meant.
EXPECTED_ORDER = [
    "Qwen3_5DenseProfile",
    "Qwen3_5Profile",
    "Qwen3Profile",
    "Qwen4ExpProfile",
    "Gemma4Profile",
    "Lfm2MoeProfile",
    "MiniMaxM2Profile",
    "DeepseekV4Profile",
    "HyV3Profile",
    "LagunaProfile",
]

# Every (model_type, architectures) pair the registered profiles must agree
# about. Drawn from the domains the Python `matches()` bodies actually test:
# each profile's own model_types and architecture prefixes, plus the
# near-misses that make ordering load-bearing.
CONFIGS: list[tuple[str, list[str]]] = [
    # Qwen3 dense / MoE
    ("qwen3", ["Qwen3ForCausalLM"]),
    ("", ["Qwen3ForCausalLM"]),
    ("", ["Qwen3ForSequenceClassification"]),   # exact-match, not a prefix
    ("qwen3_moe", ["Qwen3MoeForCausalLM"]),
    ("", ["Qwen3MoeForCausalLM"]),
    # Qwen4-Exp / Qwen3.8-Flash-Next
    ("qwen4_exp", ["Qwen4ExpForConditionalGeneration"]),
    ("qwen4_exp_text", ["Qwen4ExpForCausalLM"]),
    ("", ["Qwen4ExpTextModel"]),
    # Qwen3.5 / 3.6 MoE
    ("qwen3_5_moe", ["Qwen3_5MoeForConditionalGeneration"]),
    ("qwen3_5_moe_text", ["Qwen3_5MoeForCausalLM"]),
    ("qwen3_6_moe", ["Qwen3_6MoeForCausalLM"]),
    ("", ["Qwen3_5MoeTextModel"]),
    ("", ["Qwen3.6MoeForCausalLM"]),
    # Qwen3.5 / 3.6 dense — including the case the drifted spec got wrong
    ("qwen3_5", ["Qwen3_5ForConditionalGeneration"]),
    ("qwen3_6", ["Qwen3_6ForCausalLM"]),
    ("qwen3_6", ["Qwen3_6MoeForCausalLM"]),
    ("qwen3_5", ["Qwen3_5MoeForCausalLM"]),
    ("", ["Qwen3.5ForCausalLM"]),
    # the rest of the family
    ("gemma4", ["Gemma4ForConditionalGeneration"]),
    ("gemma4_text", ["Gemma4ForCausalLM"]),
    ("gemma4_unified", ["Gemma4UnifiedForConditionalGeneration"]),
    ("lfm2_moe", ["Lfm2MoeForCausalLM"]),
    ("lfm2-moe", ["Lfm2MoeForCausalLM"]),
    ("minimax_m2", ["MiniMaxM2ForCausalLM"]),
    ("minimax_m2.7", ["MiniMaxM2ForCausalLM"]),
    ("minimax-m2", ["MiniMax-M2ForCausalLM"]),
    ("deepseek_v4", ["DeepseekV4ForCausalLM"]),
    ("deepseek-v4", ["DeepSeek-V4ForCausalLM"]),
    ("hy_v3", ["HYV3ForCausalLM"]),
    ("", ["HYV3MTPForCausalLM"]),
    ("laguna", ["LagunaForCausalLM"]),
    ("", ["LagunaSForCausalLM"]),
    # nobody's
    ("llama", ["LlamaForCausalLM"]),
    ("", []),
    ("mistral", ["MistralForCausalLM"]),
]
CONFIG_IDS = [f"{mt or '-'}|{','.join(a) or '-'}" for mt, a in CONFIGS]


def _spec_for(cls) -> object | None:
    return load_structure_spec(cls().name)


# ------------------------------------------------------------------ vocabulary


def test_match_vocabulary_rejects_unknown_keys():
    """A typo in a `match` block must fail at parse time. The whole reason
    `qwen3_5_dense.json` drifted is that nothing ever read the field."""
    with pytest.raises(ValueError, match="unsupported model-structure match"):
        SpecMatch.from_dict({"model_types": ["qwen3"]})


def test_exclude_vetoes_a_model_type_hit():
    """The negative list is a veto, not a filter on the arch clause: the dense
    profile must not claim `model_type=qwen3_6` when the arch says Moe."""
    m = SpecMatch.from_dict({
        "model_type": ["qwen3_6"],
        "architectures": ["Qwen3_6For*"],
        "architectures_exclude": ["*Moe*"],
    })
    assert m.claims("qwen3_6", ["Qwen3_6ForCausalLM"])
    assert not m.claims("qwen3_6", ["Qwen3_6MoeForCausalLM"])


def test_architectures_are_globs_and_case_sensitive():
    m = SpecMatch.from_dict({"architectures": ["Qwen3ForCausalLM"]})
    assert m.claims("", ["Qwen3ForCausalLM"])
    assert not m.claims("", ["Qwen3ForSequenceClassification"])
    assert not m.claims("", ["qwen3forcausallm"])


def test_undeclared_match_claims_nothing():
    assert not SpecMatch.from_dict({}).claims("qwen3", ["Qwen3ForCausalLM"])
    assert not SpecMatch.from_dict({}).declared


# -------------------------------------------------------------- the strict gate


@pytest.mark.parametrize("cls", PROFILE_CLASSES, ids=[c.__name__ for c in PROFILE_CLASSES])
@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_spec_verdict_equals_python_verdict(cls, cfg):
    """The R8 mitigation, stated as a test.

    Every registered profile, every representative config: `spec.match` and
    the Python `matches()` must return the same boolean. A red here means one
    of the two has drifted — which is exactly how `qwen3_5_dense.json` lost
    its Moe exclusion while nothing read the field.
    """
    spec = _spec_for(cls)
    assert spec is not None, f"{cls.__name__} has no structure spec"
    model_type, archs = cfg
    python_verdict = bool(cls.matches(model_type, archs))
    spec_verdict = bool(SpecMatchProfile(spec).claims(model_type, archs))
    assert spec_verdict == python_verdict, (
        f"{cls.__name__} on (model_type={model_type!r}, archs={archs}): "
        f"python={python_verdict} spec={spec_verdict}"
    )


def test_spec_resolution_matches_python_resolution():
    """Whole-registry equivalence: resolving each config through the spec
    verdicts in priority order picks the same profile the live registry does.
    Per-profile agreement does not imply this on its own — ordering does."""
    spec_by_name = {spec.id: spec for spec in iter_structure_specs()}
    ordered = sorted(
        PROFILE_CLASSES,
        key=lambda c: (int(c.priority), PROFILE_CLASSES.index(c)),
    )
    for model_type, archs in CONFIGS:
        live = _registry.profile_from_config(
            {"model_type": model_type, "architectures": archs})
        spec_pick = None
        for cls in ordered:
            spec = spec_by_name.get(cls().name)
            if spec is not None and SpecMatchProfile(spec).claims(model_type, archs):
                spec_pick = cls().name
                break
        expected = None if type(live).__name__ == "DefaultProfile" else live.name
        assert spec_pick == expected, (
            f"(model_type={model_type!r}, archs={archs}): live={expected!r} "
            f"spec-only={spec_pick!r}")


# ------------------------------------------------------------------- priorities


def test_priority_order_equals_registered_order():
    """`priority` replaced the comment-encoded order at `registry.py`. Sorting
    by it must reproduce the list literal exactly, or detection changed."""
    assert [c.__name__ for c in PROFILE_CLASSES] == EXPECTED_ORDER
    by_priority = sorted(
        range(len(PROFILE_CLASSES)),
        key=lambda i: (int(PROFILE_CLASSES[i].priority), i),
    )
    assert by_priority == list(range(len(PROFILE_CLASSES))), (
        "priority order diverges from _REGISTERED order: "
        f"{[(c.__name__, c.priority) for c in PROFILE_CLASSES]}")
    assert [c.__name__ for c in _registry.detection_order()] == EXPECTED_ORDER


def test_every_profile_declares_its_own_priority():
    """Inherited priorities are a trap: Qwen3_5DenseProfile subclasses
    Qwen3_5Profile, so a missing declaration silently ties a subset to its
    superset."""
    for cls in PROFILE_CLASSES:
        assert "priority" in vars(cls), (
            f"{cls.__name__} inherits its detection priority instead of "
            "declaring one")


def test_spec_priority_agrees_with_python_priority():
    """The spec must carry the same priority as its Python profile — that is
    what lets the Python body be deleted later without changing detection."""
    for cls in PROFILE_CLASSES:
        spec = _spec_for(cls)
        assert spec is not None, f"{cls.__name__} has no structure spec"
        assert spec.priority == cls.priority, (
            f"{cls.__name__}: python priority {cls.priority} != spec "
            f"{spec.priority} in specs/{cls().name}.json")
        assert spec.priority != SPEC_DEFAULT_PRIORITY, (
            f"specs/{cls().name}.json declares no `priority`; it would "
            "resolve after every Python profile once its class is deleted")


def test_third_party_registration_still_wins():
    """`register_profile` documents insert-at-front override. The default
    `ModelProfile.priority = 0` is what keeps that true under priority
    ordering."""
    from prismaquant.model_profiles.base import ModelProfile

    class _ThirdParty(ModelProfile):
        @classmethod
        def matches(cls, model_type, architectures):
            return model_type == "qwen3_5_moe"

        @property
        def name(self):
            return "third_party_test"

    assert _ThirdParty.priority == 0
    _registry.register_profile(_ThirdParty)
    try:
        resolved = _registry.profile_from_config(
            {"model_type": "qwen3_5_moe",
             "architectures": ["Qwen3_5MoeForConditionalGeneration"]})
        assert isinstance(resolved, _ThirdParty)
        assert _registry.detection_order()[0] is _ThirdParty
    finally:
        _registry._REGISTERED.remove(_ThirdParty)
        _registry._REGISTRY_GENERATION += 1


# ------------------------------------------------------------ spec-only profile


def test_spec_only_profile_is_built_for_an_unclaimed_spec(tmp_path, monkeypatch):
    """The extension point itself: a spec no Python profile claims becomes a
    `SpecMatchProfile` in the detection order, ahead of `DefaultProfile`."""
    from prismaquant.model_profiles import structure as _structure

    spec = _structure.ModelStructureSpec.from_dict({
        "schema": _structure.SCHEMA,
        "id": "zeropython_test_arch",
        "priority": 500,
        "match": {"model_type": ["zeropython"],
                  "architectures": ["ZeroPythonFor*"]},
        "fused_groups": [{
            "target": "self_attn.qkv_proj",
            "members": ["self_attn.q_proj", "self_attn.k_proj",
                        "self_attn.v_proj"],
        }],
    })
    real = _structure.iter_structure_specs

    monkeypatch.setattr(
        _structure, "iter_structure_specs", lambda: (*real(), spec))
    monkeypatch.setattr(_registry, "_DETECTION_ORDER_CACHE", None)
    monkeypatch.setattr(
        _registry, "_REGISTRY_GENERATION", _registry._REGISTRY_GENERATION + 1)

    resolved = _registry.profile_from_config(
        {"model_type": "zeropython", "architectures": ["ZeroPythonForCausalLM"]})
    assert isinstance(resolved, SpecMatchProfile)
    assert resolved.name == "zeropython_test_arch"
    # It is a fully functional tier-2 profile, not just a matcher.
    assert resolved.fused_sibling_group("model.layers.0.self_attn.q_proj") == (
        "model.layers.0.self_attn.qkv_proj")
    # ...and it does not shadow any registered Python profile.
    still = _registry.profile_from_config(
        {"model_type": "qwen3_5_moe",
         "architectures": ["Qwen3_5MoeForConditionalGeneration"]})
    assert type(still).__name__ == "Qwen3_5Profile"

    monkeypatch.undo()
    _registry._DETECTION_ORDER_CACHE = None
    _registry._REGISTRY_GENERATION += 1


def test_python_profile_wins_over_a_same_named_spec():
    """Every shipped spec is claimed by a Python profile today, so no
    SpecMatchProfile may appear in the live detection order. This is what
    makes landing the reader a no-op for shipped models."""
    assert not [c for c in _registry.detection_order()
                if isinstance(c, SpecMatchProfile)]
