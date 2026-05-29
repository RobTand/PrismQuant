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

Registered profiles are consulted in registration order; the first one
whose `.matches()` returns True wins. `DefaultProfile` is the terminal
fallback when nothing matches.
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
from .qwen3_moe import Qwen3MoeProfile

# MiniMaxM2Profile: re-imported from its live mirror after the 2026-04-24
# session's Phase-3 archive move. The profile is still tracked under
# archive/minimax_m2p7/ as its canonical home; this live import enables
# allocator Pareto runs without uprooting the archive commit.
from .minimax_m2 import MiniMaxM2Profile
from .deepseek_v4 import DeepseekV4Profile


_REGISTERED: list[type[ModelProfile]] = [
    Qwen3_5DenseProfile,  # must precede Qwen3_5Profile (dense is a subset)
    Qwen3_5Profile,
    Qwen3MoeProfile,  # must precede Qwen3Profile (MoE model_type includes qwen3)
    Qwen3Profile,  # original Qwen3 (dense, no MoE, no MTP) — after the 3.5 siblings
    Gemma4Profile,
    Lfm2MoeProfile,
    MiniMaxM2Profile,
    DeepseekV4Profile,
]


def register_profile(cls: type[ModelProfile]) -> None:
    """Register a new ModelProfile subclass for auto-detection.

    Profiles are consulted in registration order. Register earlier than
    built-in profiles to override them."""
    if cls not in _REGISTERED:
        _REGISTERED.insert(0, cls)


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


def _resolve(model_type: str, archs: list[str]) -> ModelProfile:
    """Walk registered profile classes, instantiate the first match."""
    for cls in _REGISTERED:
        try:
            if cls.matches(model_type, archs):
                inst = cls()
                # Some profiles need to register vendored modeling code
                # with transformers before the model loads. Defer to
                # the profile method (refactor #32) so callers don't
                # need to know the architecture-specific bootstrap.
                try:
                    inst.register_vendored_modeling()
                except Exception:
                    # Don't let a vendoring failure block profile
                    # detection — surface via the eventual model load
                    # error instead.
                    pass
                return inst
        except Exception:
            continue
    return DefaultProfile(architectures=archs)
