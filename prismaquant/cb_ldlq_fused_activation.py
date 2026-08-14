"""Fused activation policy for gate/up concatenation.

This policy is versioned so packed activation evidence, projection identities
and manifests can stamp the exact semantics. Simple concatenation along rows
in profile projection order, requiring equal row counts so the mean(square(cat))
exactly matches unweighted mean-pooled member col-weight pooling.

Version string is the public policy identifier.
"""

concat_equal_member_samples = "prismaquant.cb_ldlq_fused_activation.concat_equal_member_samples.v1"
# Alias for discoverability / import stability
FUSED_ACTIVATION_POLICY_V1 = concat_equal_member_samples


def get_packed_expert_projection_names_strict(profile, packed_proj: str) -> tuple[str, ...]:
    """Strict profile-order helper — fail-closed, no fallback.

    Profile projection order is authoritative. Any import/profile error
    fails closed — no sorted, regex, or hardcoded gate/up fallback.
    Validates returned order is nonempty.
    """
    if profile is None:
        raise ValueError(
            f"packed projection order requires profile for {packed_proj!r}, got None"
        )
    try:
        names = profile.packed_expert_projection_names(packed_proj)
    except Exception as exc:
        raise RuntimeError(
            f"profile.packed_expert_projection_names failed for {packed_proj!r}: {exc}"
        ) from exc
    if not names:
        raise ValueError(
            f"profile returned empty projection order for {packed_proj!r}"
        )
    # Normalize to tuple of str, preserve profile order
    result = tuple(str(x) for x in names)
    if not result:
        raise ValueError(
            f"profile returned empty projection order for {packed_proj!r}"
        )
    return result


__all__ = [
    "concat_equal_member_samples",
    "FUSED_ACTIVATION_POLICY_V1",
    "get_packed_expert_projection_names_strict",
]
