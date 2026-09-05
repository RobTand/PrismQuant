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

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn

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


def _qwen3_model():
    """A live model the qwen3 profile claims, for `profile_from_model` sites.

    `profile_from_config` reads only `model_type` / `architectures` off
    `model.config`, and every call site below reaches detection before it
    touches the module tree, so a bare `nn.Module` carrying a config is
    enough to exercise the refusal.
    """
    model = nn.Module()
    model.config = SimpleNamespace(**QWEN3_CONFIG)
    return model


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


# --- the same swallow one level up (issue #201, census in #202) -------------
#
# These three call sites each wrap detection in a broad `except Exception` and
# continue with a substituted profile, so before this they converted the
# refusal straight back into the silent wrong answer at their own call site.
# Each documents its tolerance as "an architecture this build does not know",
# which is a different statement from "a known architecture on a dead path".
# They are pinned here rather than in each module's own test file because the
# rule is the registry's -- where this refusal is allowed to stop -- and it has
# one home. The other 22 such call sites, across 11 modules, are censused with
# file:line in #202; several are optional-hint paths where `None` is right for
# an ABSENT profile and wrong for a DEAD one, so each needs its own judgement.


def test_the_completeness_gate_does_not_swallow_the_refusal(tmp_path, monkeypatch):
    """`artifact_completeness._detect_profile_quietly` answered `None`."""
    from prismaquant.artifact_completeness import _detect_profile_quietly

    path = _checkpoint(tmp_path, **QWEN3_CONFIG)
    assert _detect_profile_quietly(path).name == "qwen3"
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        _detect_profile_quietly(path)


def test_the_incremental_probe_shard_detection_does_not_swallow_it(
    tmp_path, monkeypatch
):
    """`incremental_probe._detect_profile_for_shards` answered `DefaultProfile`."""
    from prismaquant.incremental_probe import _detect_profile_for_shards

    path = _checkpoint(tmp_path, **QWEN3_CONFIG)
    assert _detect_profile_for_shards(path).name == "qwen3"
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        _detect_profile_for_shards(path)


def test_the_native_export_validator_does_not_swallow_it(tmp_path, monkeypatch):
    """`validate_native_export._resolve_validation_target_profile` answered `None`.

    Which then resolved the smoke's serving profile from the `default=`
    argument — a validator picking its target by fallback, on a checkpoint it
    had just been told it cannot reason about.
    """
    from prismaquant.validate_native_export import _resolve_validation_target_profile

    path = _checkpoint(tmp_path, **QWEN3_CONFIG)
    assert _resolve_validation_target_profile(path, None)
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        _resolve_validation_target_profile(path, None)


# --- the remaining 22 call sites, one module at a time (issue #202) ---------
#
# Same rule, same home: these pin where the registry's refusal is allowed to
# stop, so they live beside #201's rather than in each module's own test file.
# A site whose enclosing function cannot be reached without a GPU, a loaded
# checkpoint or a built streaming context is pinned by the tree-wide census
# test at the bottom of this file instead.


def test_rtn_staging_does_not_swallow_the_refusal(tmp_path, monkeypatch):
    """`build_rtn_cache.stage_multimodal` answered `keep_composite = False`.

    Which silently selects the text-only flatten for a family whose own
    comment says it "cannot survive" that flatten.
    """
    from prismaquant.build_rtn_cache import stage_multimodal

    path = _checkpoint(
        tmp_path, **QWEN3_CONFIG, text_config={"model_type": "qwen3"}
    )
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        stage_multimodal(path)


def test_rtn_quantizable_tensor_walk_does_not_swallow_the_refusal(monkeypatch):
    """`build_rtn_cache.iter_quantizable_tensors` answered `profile = None`.

    The profile owns which packed-MoE parameter names are quantizable, so a
    dead override silently changed the set of tensors RTN quantizes.
    """
    from prismaquant.build_rtn_cache import iter_quantizable_tensors

    model = _qwen3_model()
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        list(iter_quantizable_tensors(model))


def _dummy_shard_kwargs(**overrides):
    """Filler for a cost-shard runner's arguments.

    Every shard runner below detects the profile before it reads any of
    these, so the refusal must arrive without a real activation cache,
    format spec list or output path.
    """
    kwargs = dict(
        linear_include=".*", probe_stats={}, act_cache=None, specs=[],
        device="cpu", dtype=None, mode="unbatched", chunk_size=1,
        h_detail=None, output_path="unused", model_name="m",
        probe_path="p",
    )
    kwargs.update(overrides)
    return kwargs


def test_incremental_body_cost_shard_does_not_swallow_the_refusal(monkeypatch):
    """`_run_body_cost_shard` answered `profile = None` and sharded on."""
    from prismaquant.incremental_measure_quant_cost import _run_body_cost_shard

    ctx = SimpleNamespace(
        model=_qwen3_model(), num_layers=1, layers_prefix="model.layers."
    )
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        _run_body_cost_shard(ctx, shard_kind="body", **_dummy_shard_kwargs())


def test_incremental_visual_cost_shard_does_not_swallow_the_refusal(monkeypatch):
    """`_run_visual_cost_shard` answered `profile = None` and priced on."""
    from prismaquant.incremental_measure_quant_cost import _run_visual_cost_shard

    mm_ctx = SimpleNamespace(model=_qwen3_model(), visual_module=nn.Module())
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        _run_visual_cost_shard(
            model_path="unused",
            mm_ctx=mm_ctx,
            # A non-empty stat the include regex matches, or the runner
            # short-circuits to an empty shard before it ever detects.
            **_dummy_shard_kwargs(probe_stats={"visual.blocks.0.attn.qkv": {}}),
        )


def test_concat_merger_build_does_not_swallow_the_refusal(monkeypatch):
    """`layer_streaming._build_concat_merger` answered `None`.

    "No merger" leaves the loader unchanged, so a split-source checkpoint
    would load with its merge target unfilled.
    """
    from prismaquant.layer_streaming import _build_concat_merger

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        _build_concat_merger(_qwen3_model(), {})


def test_expert_packer_build_does_not_swallow_the_refusal(monkeypatch):
    """`layer_streaming._build_expert_packer` answered `None`.

    Which leaves a per-expert-on-disk checkpoint unpacked against packed
    live params, so the experts never load.
    """
    from prismaquant.layer_streaming import _build_expert_packer

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        _build_expert_packer(_qwen3_model(), {})


def test_packed_expert_fill_does_not_swallow_the_refusal(tmp_path, monkeypatch):
    """`layer_streaming.fill_packed_experts_from_source` answered `0`.

    #202 listed this as a candidate to keep swallowing because it returns a
    count. It is the opposite case: the function exists so packed params do
    not stay zero-initialized, so a clean `0` on a dead override reports
    exactly the silent breakage it was written to prevent.
    """
    from prismaquant.layer_streaming import fill_packed_experts_from_source

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        fill_packed_experts_from_source(_qwen3_model(), str(tmp_path))


def test_head_resident_prefixes_do_not_swallow_the_refusal(monkeypatch):
    """`layer_streaming._head_prefixes` fell back to a hardcoded guess.

    The legacy branch only knows `hc_head`, so a dead override silently
    dropped whatever head-resident prefixes the live profile declares.
    """
    from prismaquant.layer_streaming import _head_prefixes

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        _head_prefixes(_qwen3_model(), "model")


def test_cost_pass_does_not_swallow_the_refusal(monkeypatch):
    """`measure_quant_cost.run_cost_pass` answered `model_profile = None`.

    And went on to measure per-(Linear, format) cost against upstream
    modelling code, writing numbers the allocator later spends.
    """
    from prismaquant.measure_quant_cost import run_cost_pass

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        run_cost_pass(
            _qwen3_model(), None, set(), [], [], "m", "p", "cpu", None,
            "unbatched", 1, "unused",
        )


def test_perturbed_x_staging_does_not_swallow_the_refusal(tmp_path, monkeypatch):
    """`perturbed_x_cache.stage_text_only_under_work_root` answered `None`.

    Which stages the checkpoint with the hardcoded default strip-key list
    instead of the one the profile declares, so every perturbed-activation
    row is then collected from a differently-staged model.
    """
    from prismaquant.perturbed_x_cache import stage_text_only_under_work_root

    path = _checkpoint(tmp_path, **QWEN3_CONFIG)
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        stage_text_only_under_work_root(path, tmp_path / "work")


def test_production_cache_fill_does_not_swallow_the_refusal(monkeypatch):
    """`production_weight_cache.fill_production_weight_cache` answered `None`.

    This is the render that produces the bytes an export later ships, and
    `None` also drops the profile's pinned names, so components the profile
    forbids quantizing would be quantized.
    """
    import torch

    from prismaquant.production_weight_cache import fill_production_weight_cache

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        fill_production_weight_cache(
            _qwen3_model(),
            torch.zeros(1, 4, dtype=torch.long),
            ["a"],
            progress=False,
        )


def test_probe_staging_does_not_swallow_the_refusal(tmp_path, monkeypatch):
    """`sensitivity_probe.stage_text_only` answered `profile = None`.

    The twin of the perturbed-x staging site: the hardcoded strip-key list
    stages a different config than the profile declares, and every probe
    statistic gathered afterwards describes that wrong staging.
    """
    from prismaquant.sensitivity_probe import stage_text_only

    path = _checkpoint(tmp_path, **QWEN3_CONFIG)
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        stage_text_only(path)


def test_moe_structure_discovery_does_not_swallow_the_refusal(monkeypatch):
    """`sensitivity_probe.discover_moe_structure` answered `profile = None`.

    Which narrows the packed-expert projection candidates, so MoE experts go
    undiscovered and are simply never probed.
    """
    from prismaquant.sensitivity_probe import discover_moe_structure

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        discover_moe_structure(_qwen3_model())


def test_moe_router_discovery_does_not_swallow_the_refusal(monkeypatch):
    """`sensitivity_probe.discover_moe_routers` answered `profile = None`.

    Under-reporting routers also under-reports the coverage accounting built
    on them.
    """
    from prismaquant.sensitivity_probe import discover_moe_routers

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        discover_moe_routers(_qwen3_model())


def test_fisher_accumulator_does_not_swallow_the_refusal(monkeypatch):
    """`sensitivity_probe.FisherAccumulator` answered `model_profile = None`.

    Then declared a `DefaultProfile` name projection, which its own comment
    calls "an explicit declaration, not a silent degrade" — true for an
    unmatched architecture, false for a matched one on a dead path.
    """
    from prismaquant.sensitivity_probe import FisherAccumulator

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        FisherAccumulator(_qwen3_model(), [], {})


def test_fp8_rewrite_bypass_does_not_swallow_the_refusal(tmp_path, monkeypatch):
    """`streaming_model._bypass_hf_fp8_module_rewrite` answered `False`.

    #202 listed this as a possible legitimate swallow because it returns a
    bool. It is not one: `False` lets transformers run the FP8 module rewrite
    on exactly the architecture whose profile would have forbidden it.
    """
    from prismaquant.streaming_model import _bypass_hf_fp8_module_rewrite

    # The bypass question is only asked of a native-FP8 block-scaled
    # checkpoint, so the config must carry that quantization_config.
    path = _checkpoint(
        tmp_path,
        **QWEN3_CONFIG,
        quantization_config={
            "quant_method": "fp8", "weight_block_size": [128, 128],
        },
    )
    assert _bypass_hf_fp8_module_rewrite(path) in (True, False)
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        _bypass_hf_fp8_module_rewrite(path)


def test_rotary_init_does_not_swallow_the_refusal(monkeypatch):
    """`streaming_model._init_rotary_inplace` fell through to single-rope.

    The multi-layer-type architectures that override `init_rotaries` (DSv4,
    Gemma3) would silently get the single-rope path, so their per-layer-type
    rotary buffers are never registered.
    """
    from prismaquant.streaming_model import _init_rotary_inplace

    base = _qwen3_model()
    rotary = nn.Module()
    rotary.config = base.config
    rotary.compute_default_rope_parameters = lambda cfg, device: (None, None)
    base.rotary_emb = rotary

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        _init_rotary_inplace(base, "cpu", None)


def test_weight_session_does_not_swallow_the_refusal(monkeypatch):
    """`weight_session.WeightSession` answered `profile = None`.

    The profile builds the qname -> live Linear map the session swaps weights
    through, so a dead override silently builds it over a different set of
    tensors and the session reverts and re-renders the wrong ones.
    """
    from prismaquant.weight_session import WeightSession

    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "synthetic failure")
    with pytest.raises(DeadVendoredOverrideError):
        WeightSession(_qwen3_model())


# --- the rule itself, pinned once over the whole tree (issue #202) ----------


DETECTION_ENTRYPOINTS = {
    "detect_profile",
    "detect_profile_with_warning",
    "profile_from_config",
    "profile_from_model",
}


def _handler_is_broad(handler: ast.ExceptHandler) -> bool:
    """True for `except:`, `except Exception:`, `except BaseException:`."""
    node = handler.type
    if node is None:
        return True
    parts = node.elts if isinstance(node, ast.Tuple) else [node]
    return any(
        isinstance(p, ast.Name) and p.id in ("Exception", "BaseException")
        for p in parts
    )


def _handler_always_reraises(handler: ast.ExceptHandler) -> bool:
    """True when the handler cannot fall through to a substituted answer.

    `sample_parallel_probe.prepare_worker_source_cache` is the shape this
    allows: it catches broadly and re-raises as `SampleParallelProbeError(...)
    from exc`, which refuses by name and keeps the cause. A handler that always
    raises never converts the refusal into an answer, so it needs no
    `DeadVendoredOverrideError` clause of its own.
    """
    return bool(handler.body) and isinstance(handler.body[-1], ast.Raise)


def _refuses_dead_override(node: ast.Try) -> bool:
    """True when a dead override escapes this `try` instead of being answered."""
    for handler in node.handlers:
        if _handler_is_broad(handler):
            # Reached the catch-all without having refused first.
            return _handler_always_reraises(handler)
        named = handler.type
        parts = named.elts if isinstance(named, ast.Tuple) else [named]
        for part in parts:
            name = getattr(part, "id", None) or getattr(part, "attr", None)
            if name == "DeadVendoredOverrideError":
                return True
    return True  # no broad handler at all: nothing swallows it


def _swallowing_detection_sites() -> list[str]:
    """Every `file:line` where detection sits in a `try` that answers instead."""
    root = Path(__file__).resolve().parent.parent / "prismaquant"
    bad: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "archive" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try) or _refuses_dead_override(node):
                continue
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if not isinstance(sub, ast.Call):
                        continue
                    func = sub.func
                    name = getattr(func, "id", None) or getattr(func, "attr", None)
                    if name in DETECTION_ENTRYPOINTS:
                        rel = path.relative_to(root.parent)
                        bad.append(f"{rel}:{sub.lineno} ({name})")
    return sorted(set(bad))


def test_no_call_site_swallows_the_dead_override_refusal():
    """The refusal must reach the operator from every detection call site.

    #201 made `_resolve` raise `DeadVendoredOverrideError` and stopped three
    gate call sites from re-swallowing it. #202 is the rest of the census: 22
    further call sites across 11 modules each wrapped detection in a broad
    `except Exception` and continued with a substituted profile (`None`,
    `DefaultProfile()`, `keep_composite = False`, a hardcoded prefix guess, a
    count of 0, a bare `False`), which converts the refusal straight back into
    the silent wrong answer one level up.

    This is a structural pin as well as 18 behavioural ones above, because the
    rule is structural: a dead override must never be *answered*. It therefore
    also covers the four sites whose enclosing function cannot be reached
    without a GPU, a loaded checkpoint or a built streaming context
    (`aqua_activation_cost.py:661`, `build_rtn_cache.py:500`,
    `sensitivity_probe.py:3592`, `streaming_production_cache.py:1776`) — and it
    fails for a call site added next month, which no per-site test would.

    The allowed shapes are (a) `except DeadVendoredOverrideError: raise` ahead
    of the broad handler, or (b) a broad handler that always re-raises, which
    is what `sample_parallel_probe` already did correctly.
    """
    assert _swallowing_detection_sites() == []
