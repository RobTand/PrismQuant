"""The profile registry must refuse a model_type whose vendored override died.

Issue #19's real defect was silence: `register_qwen3()` returned cleanly, set
its "registered" flag, and the run then executed UPSTREAM modelling code with
no exception anywhere. `prismaquant.vendored` now verifies its own overrides
and records the dead ones — but `detect_profile` calls
`register_vendored_modeling()` inside a `try/except: pass` (correctly: a
vendoring failure must not break profile *detection*), so the recorded failure
has to be consulted explicitly or the swallow re-hides it.

These tests pin that consultation, not the vendored machinery itself.

They also pin where the refusal is allowed to STOP (issue #201). A gate that
raises inside a `try: ... except Exception: continue` is not a gate: until
`_resolve` was reordered, a recorded dead override merely demoted the profile
that matched and detection answered `DefaultProfile` — the silent wrong answer
the gate's own docstring says it refuses. So the refusal must reach the caller
of `detect_profile`, and it must arrive as its own class, so a caller can tell
"no profile matched this checkpoint" (a `DefaultProfile`, sometimes fine) from
"a profile matched and its vendored path is dead" (never fine).
"""
from __future__ import annotations

import json

import pytest

import prismaquant.vendored as vendored
from prismaquant.model_profiles.registry import (
    DeadVendoredOverrideError,
    _refuse_dead_vendored_override,
    detect_profile,
    detect_profile_with_warning,
    profile_from_config,
)


def _checkpoint(tmp_path, **config):
    """A staged checkpoint directory carrying just a `config.json`."""
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    return str(root)


QWEN3_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
}


@pytest.fixture(autouse=True)
def _clean_override_errors():
    """OVERRIDE_ERRORS is process-global; never leak a synthetic entry."""
    saved = dict(vendored.OVERRIDE_ERRORS)
    vendored.OVERRIDE_ERRORS.clear()
    vendored.OVERRIDE_ERRORS.update(saved)
    yield
    vendored.OVERRIDE_ERRORS.clear()
    vendored.OVERRIDE_ERRORS.update(saved)


def test_gate_is_inert_when_the_override_is_healthy():
    # On the reference box the qwen3 override resolves, so nothing is recorded
    # and the gate must not fire — this is the common path and it must stay
    # free of false positives.
    vendored.OVERRIDE_ERRORS.clear()
    assert _refuse_dead_vendored_override("qwen3") is None
    assert _refuse_dead_vendored_override("llama") is None


def test_gate_raises_for_a_recorded_dead_override():
    vendored.OVERRIDE_ERRORS["qwen3"] = (
        "synthetic: AutoModelForCausalLM.register no-op'd (config __module__ "
        "filter) and the config shim could not be installed either"
    )
    with pytest.raises(RuntimeError) as exc:
        _refuse_dead_vendored_override("qwen3")
    msg = str(exc.value)
    # The message must say WHICH model_type, and that the consequence is
    # running upstream code — the whole point is that this is otherwise silent.
    assert "qwen3" in msg
    assert "UPSTREAM" in msg
    # and it must carry the vendored layer's own detail rather than replacing it
    assert "config __module__" in msg


def test_gate_is_scoped_to_the_failing_model_type():
    """One dead override must not block unrelated architectures."""
    vendored.OVERRIDE_ERRORS["qwen3"] = "synthetic failure"
    assert _refuse_dead_vendored_override("gemma4") is None
    assert _refuse_dead_vendored_override("deepseek_v4") is None
    with pytest.raises(RuntimeError):
        _refuse_dead_vendored_override("qwen3")


def test_gate_survives_a_missing_vendored_package(monkeypatch):
    """A tree without the vendored package must still detect profiles.

    The gate is a safety net, not a dependency: if `prismaquant.vendored`
    cannot be imported at all there is no override to be silently wrong about.
    """
    import builtins

    real_import = builtins.__import__

    def _no_vendored(name, *args, **kwargs):
        if name == "prismaquant.vendored" or name.startswith(
                "prismaquant.vendored."):
            raise ImportError("synthetic: vendored package absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_vendored)
    assert _refuse_dead_vendored_override("qwen3") is None


# --- where the refusal is allowed to stop (issue #201) ---------------------


def test_detect_profile_refuses_a_dead_override_instead_of_defaulting(
    tmp_path, monkeypatch
):
    """The whole point of the gate: `detect_profile` must not answer at all.

    Before #201 this returned `DefaultProfile` — `_resolve` ran the gate inside
    its per-candidate `except Exception: continue`, which ate the refusal and
    then walked on to the terminal fallback.
    """
    path = _checkpoint(tmp_path, **QWEN3_CONFIG)
    assert detect_profile(path).name == "qwen3"  # healthy path, for contrast

    monkeypatch.setitem(
        vendored.OVERRIDE_ERRORS,
        "qwen3",
        "synthetic: AutoModelForCausalLM.register no-op'd",
    )
    with pytest.raises(DeadVendoredOverrideError) as exc:
        detect_profile(path)
    msg = str(exc.value)
    assert "qwen3" in msg
    assert "UPSTREAM" in msg


def test_the_refusal_has_its_own_class_distinct_from_no_match(
    tmp_path, monkeypatch
):
    """"Nothing matched" and "the match is dead" must not look the same.

    An unregistered architecture legitimately resolves to `DefaultProfile`;
    a matched profile whose vendored path is dead never does. A caller that
    tolerates the first must be able to refuse the second without catching
    every `RuntimeError` detection can raise.
    """
    unknown = _checkpoint(
        tmp_path, model_type="not_an_architecture", architectures=["NopeForCausalLM"]
    )
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    # Unrelated architecture: unaffected, still the terminal fallback.
    assert type(detect_profile(unknown)).__name__ == "DefaultProfile"

    assert issubclass(DeadVendoredOverrideError, RuntimeError)
    assert not issubclass(RuntimeError, DeadVendoredOverrideError)


def test_profile_from_config_refuses_a_dead_override(monkeypatch):
    """The config-object entrypoint shares `_resolve`, so it shares the gate."""
    assert profile_from_config(dict(QWEN3_CONFIG)).name == "qwen3"
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        profile_from_config(dict(QWEN3_CONFIG))


def test_detect_profile_with_warning_does_not_swallow_the_refusal(
    tmp_path, monkeypatch
):
    """The tolerant entrypoint tolerates an unknown arch, not a dead override.

    `detect_profile_with_warning` exists so production entrypoints can keep
    running on a not-yet-registered model with a logged fallback. Its broad
    `except Exception` is the same defect one level up: a printed warning is
    not a refusal, and this failure means the run would execute upstream
    modelling code.
    """
    path = _checkpoint(tmp_path, **QWEN3_CONFIG)
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        detect_profile_with_warning(path, entrypoint="test")


def test_an_unaskable_candidate_still_falls_through(tmp_path, monkeypatch):
    """The reordering must not cost `_resolve` its candidate-walk tolerance.

    A profile whose `matches()` explodes is not a match; detection keeps
    walking, exactly as before. Only the dead-override refusal is promoted.
    """
    from prismaquant.model_profiles import registry
    from prismaquant.model_profiles.base import ModelProfile

    class ExplodingProfile(ModelProfile):
        priority = 1

        @classmethod
        def matches(cls, model_type, architectures):
            raise ValueError("synthetic: this candidate cannot be asked")

        @property
        def name(self) -> str:
            return "exploding"

    monkeypatch.setattr(registry, "_REGISTERED", [ExplodingProfile, *registry._REGISTERED])
    monkeypatch.setattr(registry, "_REGISTRY_GENERATION", registry._REGISTRY_GENERATION + 1)
    monkeypatch.setattr(registry, "_DETECTION_ORDER_CACHE", None)

    path = _checkpoint(tmp_path, **QWEN3_CONFIG)
    assert detect_profile(path).name == "qwen3"
