"""Shared policy helpers for fixed or allocator-selected LM heads.

Model profiles intentionally pin ``lm_head`` by default.  Production recipes
may nevertheless carry a measured, runtime-supported fixed head format, while
research recipes can lift the pin through ``--allow-pinned`` and let the DP
choose.  Cache and export callers must make the same decision or a quantized
head is either never rendered or is silently forced back to BF16.

This module contains only name/policy resolution.  Rendering still flows
through :class:`ProductionWeightCache`, and serialized byte accounting remains
owned by the allocator/footprint primitives.
"""
from __future__ import annotations

from collections.abc import Iterable


def parse_allow_pinned(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize the allocator's comma-separated ``--allow-pinned`` syntax."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = value
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _module_name(value: object) -> str:
    name = str(value)
    return name[:-7] if name.endswith(".weight") else name


def lm_head_aliases(profile) -> tuple[str, ...]:
    """Return recipe/checkpoint spellings that identify one profile's head.

    Most profiles use ``lm_head`` throughout.  DeepSeek-V4 is the important
    counterexample: allocator/live spelling is ``lm_head`` while the source
    checkpoint calls it ``head``.  Both must be lifted together at cache and
    export boundaries.
    """
    aliases = {"lm_head"}
    getter = getattr(profile, "lm_head_name", None)
    if callable(getter):
        try:
            aliases.add(_module_name(getter()))
        except Exception:
            pass
    return tuple(sorted(alias for alias in aliases if alias))


def is_lm_head_name(name: object, profile) -> bool:
    """Whether a qname is the profile's language-model output head."""
    module = _module_name(name)
    return any(
        module == alias or module.endswith("." + alias)
        for alias in lm_head_aliases(profile)
    )


def allow_pinned_lifts_name(
    name: object,
    allow_pinned: str | Iterable[str] | None,
) -> bool:
    """Match the allocator's historical ``token in qname`` semantics."""
    module = _module_name(name)
    return any(token in module for token in parse_allow_pinned(allow_pinned))


def allow_pinned_lifts_lm_head(
    profile,
    allow_pinned: str | Iterable[str] | None,
) -> bool:
    """Whether an allow token lifts the head under any profile alias.

    Alias-in-token handles an explicitly qualified spelling such as
    ``language_model.lm_head`` while retaining the allocator's normal
    token-in-name behavior.
    """
    tokens = parse_allow_pinned(allow_pinned)
    return any(
        token in alias or alias in token
        for token in tokens
        for alias in lm_head_aliases(profile)
    )


def remaining_profile_pins(
    profile,
    *,
    allow_pinned: str | Iterable[str] | None = None,
    fixed_lm_head_quantized: bool = False,
) -> tuple[str, ...]:
    """Profile pins cache/export must still keep at source precision.

    A fixed quantized head lifts every alias for that one structural tensor.
    Other profile pins remain intact.  ``allow_pinned`` independently lifts
    research/DP-selected names with the allocator's substring contract.
    """
    lift_head = bool(fixed_lm_head_quantized) or allow_pinned_lifts_lm_head(
        profile, allow_pinned
    )
    remaining: list[str] = []
    for raw_pin in profile.pinned_names():
        pin = _module_name(raw_pin)
        if allow_pinned_lifts_name(pin, allow_pinned):
            continue
        if lift_head and is_lm_head_name(pin, profile):
            continue
        remaining.append(pin)
    return tuple(dict.fromkeys(remaining))
