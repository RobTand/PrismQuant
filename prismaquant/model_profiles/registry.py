"""Profile auto-detection + manual registration.

Usage:

    from prismaquant.model_profiles import detect_profile
    profile = detect_profile("/path/to/Qwen3.6-35B-A3B")
    # profile is a Qwen3_5Profile instance.

External architectures can register their own profile at runtime:

    from prismaquant.model_profiles import register_profile, ModelProfile

    class MyArchProfile(ModelProfile):
        ...

    register_profile(MyArchProfile)

Registered profiles are consulted in `ModelProfile.priority` order (lower
first, ties broken by registration order); the first one whose `.matches()`
returns True wins. `DefaultProfile` is the terminal fallback when nothing
matches.

Two kinds of profile take part in detection:

  - **Python profiles** in `_REGISTERED`, matched by `cls.matches()`.
  - **Spec-only profiles** — a `SpecMatchProfile` per `specs/<id>.json` whose
    `id` no registered Python profile claims, matched by its declarative
    `match` block. This is the extension point that makes a pure-JSON
    architecture possible; while a Python profile of the same name exists it
    wins outright, so adding a spec never silently re-routes a shipped model.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import ModelProfile
from .default import DefaultProfile
from .gemma4 import Gemma4Profile
from .lfm2_moe import Lfm2MoeProfile
from .qwen3 import Qwen3Profile
from .qwen3_5 import Qwen3_5Profile
from .qwen3_5_dense import Qwen3_5DenseProfile
from .qwen4_exp import Qwen4ExpProfile

# MiniMaxM2Profile: re-imported from its live mirror after the 2026-04-24
# session's Phase-3 archive move. The profile is still tracked under
# archive/minimax_m2p7_2026-04-24/ as its canonical home; this live import enables
# allocator Pareto runs without uprooting the archive commit.
from .minimax_m2 import MiniMaxM2Profile
from .deepseek_v4 import DeepseekV4Profile
from .hy_v3 import HyV3Profile
from .laguna import LagunaProfile


# Detection order is load-bearing — subsets must precede supersets, and
# getting it wrong silently re-routes a shipped model to another profile. It
# used to be encoded in this list's *order* plus two comments; it is now
# encoded in each profile's `priority` class attribute (lower = consulted
# first), which survives being read from a spec file after the Python body is
# deleted. The comments below record why each pair is ordered as it is; the
# numbers are the contract, and `tests/test_spec_match_profile.py` asserts that
# priority order still equals this list's literal order.
_REGISTERED: list[type[ModelProfile]] = [
    Qwen3_5DenseProfile,  # 100 — must precede Qwen3_5Profile (dense is a subset)
    Qwen3_5Profile,       # 110
    Qwen3Profile,         # 120 — original Qwen3 dense + routed MoE
    Qwen4ExpProfile,      # 130 — Qwen3.8-Flash-Next / qwen4_exp
    Gemma4Profile,        # 140
    Lfm2MoeProfile,       # 150
    MiniMaxM2Profile,     # 160
    DeepseekV4Profile,    # 170
    HyV3Profile,          # 180
    LagunaProfile,        # 190
]

# Bumped whenever `_REGISTERED` changes, so the derived detection order (which
# folds in spec-only profiles) can be cached without going stale.
_REGISTRY_GENERATION = 0
_DETECTION_ORDER_CACHE: tuple[int, tuple] | None = None


def register_profile(cls: type[ModelProfile]) -> None:
    """Register a new ModelProfile subclass for auto-detection.

    Profiles are consulted in `priority` order (lower first). A profile that
    does not declare one inherits `ModelProfile.priority = 0`, which is ahead
    of every built-in (100+) — so the historical "register earlier than
    built-in profiles to override them" contract holds unchanged."""
    global _REGISTRY_GENERATION
    if cls not in _REGISTERED:
        _REGISTERED.insert(0, cls)
        _REGISTRY_GENERATION += 1


def _refuse_dead_vendored_override(model_type: str) -> None:
    """Fail if this model_type's vendored-modeling override is known dead.

    `register_vendored_modeling()` is called inside a `try/except: pass` above,
    because a vendoring failure must not break profile *detection*. But the
    failure mode that swallow hides is not an exception — it is a registration
    that silently no-ops (issue #19), after which the load succeeds against
    UPSTREAM modelling code and nothing ever raises. `prismaquant.vendored`
    verifies its own overrides and records the dead ones, so consult that
    rather than trusting the absence of an exception: running a probe on the
    wrong modelling path is exactly the silent-wrong-answer this project
    refuses to ship.
    """
    try:
        from prismaquant.vendored import OVERRIDE_ERRORS
    except Exception:  # noqa: BLE001 - vendored package absent entirely
        return
    detail = OVERRIDE_ERRORS.get(str(model_type))
    if detail:
        raise RuntimeError(
            f"vendored modelling override for model_type={model_type!r} did "
            f"not take effect, so a load here would silently run UPSTREAM "
            f"modelling code instead of the vendored copy:\n{detail}"
        )


def detect_profile(model_path: str) -> ModelProfile:
    """Pick the right ModelProfile for a checkpoint directory.

    Reads `config.json`, walks registered profiles, returns the first
    whose `.matches()` returns True. Falls back to `DefaultProfile` if
    nothing matches."""
    cfg_path = Path(model_path) / "config.json"
    model_type = ""
    archs: list[str] = []
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type") or ""
            archs = list(cfg.get("architectures") or [])
        except (json.JSONDecodeError, OSError):
            pass
    return _resolve(model_type, archs)


def detect_profile_with_warning(
    model_path: str,
    *,
    entrypoint: str,
) -> ModelProfile:
    """Detect a model profile and make DefaultProfile fallback observable.

    This preserves the historical fallback behavior for production entrypoints
    that intentionally keep running on vanilla or not-yet-registered models,
    but it prevents architecture-specific fused/MoE checks from disappearing
    without any log signal.
    """
    reason = ""
    try:
        profile = detect_profile(model_path)
    except Exception as exc:
        reason = f"detect_profile raised {type(exc).__name__}: {exc}"
        profile = DefaultProfile()

    if isinstance(profile, DefaultProfile):
        cfg_path = Path(model_path) / "config.json"
        if not reason:
            if not cfg_path.exists():
                reason = f"{cfg_path} missing"
            else:
                reason = "architecture unregistered or config unreadable"
        print(
            f"[{entrypoint}] WARNING: resolved DefaultProfile for "
            f"{model_path!r} ({reason}). Architecture-specific fused-sibling, "
            "packed-MoE, pinned-name, and source-passthrough rules are not "
            "available; pass the correct local model path or register a "
            "ModelProfile if this is not a vanilla transformer.",
            flush=True,
        )
    return profile


def profile_from_config(cfg) -> ModelProfile:
    """Pick a ModelProfile from a (possibly already-loaded) HF config
    object or dict. Useful for consumers that already hold the model
    (e.g. `_init_rotary_inplace`) and don't have `model_path`."""
    if cfg is None:
        return DefaultProfile(architectures=[])
    if isinstance(cfg, dict):
        model_type = cfg.get("model_type") or ""
        archs = list(cfg.get("architectures") or [])
    else:
        model_type = getattr(cfg, "model_type", "") or ""
        archs = list(getattr(cfg, "architectures", []) or [])
    return _resolve(model_type, archs)


def profile_from_model(model) -> ModelProfile:
    """Pick a ModelProfile from a live transformers model. Reads
    `model.config` and dispatches via `profile_from_config`."""
    return profile_from_config(getattr(model, "config", None))


def _python_profile_names() -> set[str]:
    """Names claimed by registered Python profiles (`specs/<name>.json` keys)."""
    names: set[str] = set()
    for cls in _REGISTERED:
        try:
            names.add(cls().name)
        except Exception:
            continue
    return names


def detection_order() -> tuple:
    """Every detection candidate, ordered by priority (lower first).

    Entries are either a `ModelProfile` subclass (matched via `cls.matches()`)
    or a `SpecMatchProfile` instance (matched via its declarative `match`
    block). Ties keep `_REGISTERED` order, and Python profiles precede
    spec-only ones at equal priority: an executable claim is the more specific
    statement, and this keeps a spec that copies its Python profile's priority
    from reordering detection.
    """
    global _DETECTION_ORDER_CACHE
    cached = _DETECTION_ORDER_CACHE
    if cached is not None and cached[0] == _REGISTRY_GENERATION:
        return cached[1]

    from .spec_profile import SpecMatchProfile
    from .structure import iter_structure_specs

    entries: list[tuple[int, int, int, object]] = []
    for index, cls in enumerate(_REGISTERED):
        entries.append((int(getattr(cls, "priority", 0)), 0, index, cls))

    claimed = _python_profile_names()
    try:
        specs = iter_structure_specs()
    except Exception:
        specs = ()
    for index, spec in enumerate(specs):
        if spec.id in claimed or not spec.match.declared:
            continue
        entries.append((int(spec.priority), 1, index, SpecMatchProfile(spec)))

    entries.sort(key=lambda item: item[:3])
    order = tuple(item[3] for item in entries)
    _DETECTION_ORDER_CACHE = (_REGISTRY_GENERATION, order)
    return order


def _claims(candidate, model_type: str, archs: list[str]) -> bool:
    if isinstance(candidate, type):
        return bool(candidate.matches(model_type, archs))
    return bool(candidate.claims(model_type, archs))


def _new_instance(candidate) -> ModelProfile:
    """Fresh profile instance per resolution (lazy caches are not shared)."""
    if isinstance(candidate, type):
        return candidate()
    return type(candidate)(candidate.spec)


def _resolve(model_type: str, archs: list[str]) -> ModelProfile:
    """Walk detection candidates in priority order, build the first match."""
    for cls in detection_order():
        try:
            if _claims(cls, model_type, archs):
                inst = _new_instance(cls)
                # Hand the profile what the checkpoint declared. A family can
                # cover several serving classes (a multimodal wrapper and its
                # text-only carve-out) whose namespaces differ, and only the
                # declaration distinguishes them. Profiles that ignore it are
                # unaffected: the stash is inert unless read.
                inst.declare_config(model_type, archs)
                # Some profiles need to register vendored modeling code
                # with transformers before the model loads. Defer to
                # the profile method (refactor #32) so callers don't
                # need to know the architecture-specific bootstrap.
                try:
                    inst.register_vendored_modeling()
                except Exception:
                    # Don't let a vendoring failure block profile
                    # DETECTION — but do not lose it either. The old
                    # comment here assumed "the eventual model load
                    # error" would surface it; that reasoning only holds
                    # for a failure that raises at load time. The failure
                    # mode this swallow actually hides is the opposite
                    # one (issue #19): a registration that silently
                    # no-ops, after which the load succeeds against the
                    # WRONG modelling code and nothing ever raises.
                    # `prismaquant.vendored` records those in
                    # OVERRIDE_ERRORS; the gate below refuses to hand
                    # back a profile whose vendored path is known dead.
                    pass
                _refuse_dead_vendored_override(model_type)
                return inst
        except Exception:
            continue
    return DefaultProfile(architectures=archs)
