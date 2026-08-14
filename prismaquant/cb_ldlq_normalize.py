"""Python counterpart to cb_ldlq_normalize.sh — single truth table for LDLQ scope/legacy.

This module is the authoritative Python normalization for PRISMAQUANT_CB_LDLQ
and PRISMAQUANT_CB_LDLQ_SCOPE. Shell uses cb_ldlq_normalize.sh; Python must
use this module. Any inline truth-table in allocator or footprint must import
from here to stay consistent.
"""

from __future__ import annotations

from typing import Mapping


def normalize_ldlq_vars(
    raw_legacy: str | None,
    raw_scope: str | None,
) -> tuple[str, str]:
    """Normalize (legacy, scope) pair to canonical (legacy_str, scope_str).

    Inputs are raw strings as from CLI or env (None means unset). Empty or
    whitespace-only scope is treated as unset. Returns (PRISMAQUANT_CB_LDLQ,
    PRISMAQUANT_CB_LDLQ_SCOPE) as canonical "0"/"1" and "none"/"nvfp4"/"all".

    Truth table (scope authoritative):
      neither set -> 0 none
      legacy 0, no scope -> 0 none
      legacy 1, no scope -> 1 all
      scope nvfp4, no legacy -> 1 nvfp4
      scope none, no legacy -> 0 none
      both set -> legacy must equal (scope != none)

    Raises ValueError on invalid or inconsistent explicit pair.
    """
    # Treat empty scope as unset
    scope_clean = raw_scope.strip().lower() if isinstance(raw_scope, str) and raw_scope.strip() != "" else None
    if scope_clean is not None and scope_clean not in {"none", "nvfp4", "all"}:
        raise ValueError(f"PRISMAQUANT_CB_LDLQ_SCOPE must be one of none/nvfp4/all, got {raw_scope!r}")

    # Normalize legacy to 0/1 if present. Empty or whitespace-only is unset.
    legacy_clean: str | None = None
    if raw_legacy is not None:
        s = str(raw_legacy).strip().lower()
        if s == "":
            legacy_clean = None
        elif s in {"1", "true", "yes", "on"}:
            legacy_clean = "1"
        elif s in {"0", "false", "no", "off"}:
            legacy_clean = "0"
        else:
            raise ValueError(f"PRISMAQUANT_CB_LDLQ must be 0 or 1, got {raw_legacy!r}")

    was_legacy = legacy_clean is not None
    was_scope = scope_clean is not None

    if not was_legacy and not was_scope:
        return "0", "none"
    if was_legacy and not was_scope:
        return (legacy_clean, "all" if legacy_clean == "1" else "none")  # type: ignore[return-value]
    if not was_legacy and was_scope:
        # scope determines legacy
        return ("1" if scope_clean != "none" else "0", scope_clean)  # type: ignore[return-value]
    # both set
    expected = "1" if scope_clean != "none" else "0"
    if legacy_clean != expected:
        raise ValueError(
            f"PRISMAQUANT_CB_LDLQ={raw_legacy!r} inconsistent with PRISMAQUANT_CB_LDLQ_SCOPE={raw_scope!r}"
        )
    return legacy_clean, scope_clean  # type: ignore[return-value]


def resolve_from_cli_and_env(
    cli_legacy: str | None,
    cli_scope: str | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve scope with CLI > env > legacy precedence, using normalize_ldlq_vars.

    CLI values win when not None/empty; otherwise env is consulted. Returns
    canonical (legacy_str, scope_str).
    """
    import os

    if environ is None:
        environ = os.environ
    # CLI scope empty string treated as unset
    cli_scope_clean = cli_scope.strip().lower() if isinstance(cli_scope, str) and cli_scope.strip() != "" else None
    cli_legacy_clean = cli_legacy if cli_legacy is not None and str(cli_legacy).strip() != "" else None

    # If CLI provides either, use CLI pair (env ignored for scope/legacy)
    if cli_scope_clean is not None or cli_legacy_clean is not None:
        # If CLI scope is set, use it; else None
        # If CLI legacy is set, use it; else None
        raw_legacy = cli_legacy_clean if cli_legacy_clean is not None else None
        raw_scope = cli_scope if cli_scope_clean is not None else None
        # But if CLI scope is set and legacy not, we still need to consider env legacy? No, CLI scope alone should determine.
        # The helper normalize will handle scope-only case.
        # However if CLI scope is set and CLI legacy is None, we should not fall back to env legacy.
        # So we directly normalize the CLI pair.
        # Special: CLI scope set but legacy None -> normalize will infer legacy.
        # CLI legacy set but scope None -> normalize will infer scope.
        # Both set -> check consistency.
        # To handle CLI scope + env legacy fallback correctly, we should only use CLI values,
        # not mix CLI scope with env legacy.
        try:
            return normalize_ldlq_vars(raw_legacy, raw_scope)
        except ValueError as exc:
            raise ValueError(f"[allocator] {exc}") from None

    # No CLI — use env
    raw_legacy_env = environ.get("PRISMAQUANT_CB_LDLQ")
    raw_scope_env = environ.get("PRISMAQUANT_CB_LDLQ_SCOPE")
    # Empty env scope already treated as unset in normalize
    return normalize_ldlq_vars(raw_legacy_env, raw_scope_env)
