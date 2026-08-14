"""Activation-cache adapter for production CB LDLQ assignment.

This is a reader over the established probe cache, not a second cache. Dense
and per-expert Linears reuse their captured input rows directly; packed-MoE
stages use the existing checkpoint replay in :mod:`prismaquant.moe_imatrix`.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import torch


_ACT_FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")

# Fused activation evidence policy — versioned and stamped in evidence/identities.
# Importing from the dedicated module keeps the string single-sourced.
from .cb_ldlq_fused_activation import (
    concat_equal_member_samples as FUSED_ACTIVATION_POLICY_V1,
    get_packed_expert_projection_names_strict,
)

# Back-compat alias for older import paths
FUSED_ACTIVATION_POLICY = FUSED_ACTIVATION_POLICY_V1


class NoObservedExpertRowsError(ValueError):
    """Typed failure: routed replay has no observed expert rows to pool."""


def fill_empty_expert_activation_rows(
    rows: tuple[torch.Tensor, ...],
    *,
    qname: str,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
    """Apply the declared cold-expert prior to sparse routed LDLQ rows.

    A bounded module-level activation reservoir can legitimately miss experts
    that the full calibration forward routed. Observed experts retain their
    exact rows; empty slices receive the pooled routed rows from the same
    layer/projection. This is the activation analogue of the imatrix
    layer-routed-mean neutral prior and never fabricates a cross-layer sample.
    """
    missing = tuple(i for i, value in enumerate(rows) if not value.shape[0])
    if not missing:
        return rows, ()
    observed = [value for value in rows if value.shape[0]]
    if not observed:
        raise NoObservedExpertRowsError(
            f"{qname}: LDLQ routed replay has no observed expert rows"
        )
    widths = {int(value.shape[1]) for value in observed}
    if len(widths) != 1:
        raise ValueError(
            f"{qname}: LDLQ routed expert rows disagree on input width"
        )
    pooled = torch.cat(observed, dim=0).contiguous()
    filled = tuple(pooled if not value.shape[0] else value for value in rows)
    return filled, missing


class CBLDLQActivationLoader:
    """Load one target's Hessian rows lazily from the production act cache."""

    def __init__(
        self,
        activation_cache_dir: str | Path,
        *,
        model_dir: str | Path,
        profile,
        expert_stack_members: Mapping[
            str, Mapping[tuple[str, int], str]
        ] | None = None,
        replay_device: str | None = None,
    ) -> None:
        self.activation_cache_dir = Path(activation_cache_dir)
        self.model_dir = Path(model_dir)
        self.profile = profile
        self.expert_stack_members = dict(expert_stack_members or {})
        self.replay_device = replay_device

    def _direct(self, qname: str) -> torch.Tensor | None:
        path = self.activation_cache_dir / (
            _ACT_FNAME_SUB.sub("__", str(qname)) + ".pt"
        )
        if not path.is_file():
            return None
        blob = torch.load(path, map_location="cpu", weights_only=False)
        value = blob.get("inputs") if isinstance(blob, dict) else None
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(
                f"{qname}: LDLQ activation cache entry has no rank-2 inputs"
            )
        return value.detach().to(torch.float32).contiguous()

    def _per_expert(self, qname: str) -> tuple[torch.Tensor, ...] | None:
        members = self.expert_stack_members.get(qname)
        if not members:
            return None
        # Derive packed_proj and deterministic profile-order for concatenation.
        packed_proj = str(qname).rsplit(".", 1)[-1] if "." in str(qname) else str(qname)
        # Strict authoritative order — fail closed on any import/profile error,
        # no sorted/regex/hardcoded fallback. Delegates to neutral shared helper
        # to avoid circular import with streaming exporter.
        projections = get_packed_expert_projection_names_strict(self.profile, packed_proj)
        if not projections:
            raise ValueError(f"{qname}: profile returned empty projection order for {packed_proj!r}")
        declared_projections = {str(p) for p, _ in members.keys()}
        if set(projections) != declared_projections:
            raise ValueError(
                f"{qname}: declared member projections {sorted(declared_projections)} "
                f"!= profile order {list(projections)} for {packed_proj!r}"
            )
        # Single-member (down_proj): preserve missing experts as empty placeholders
        # instead of raising; partial vs full missing is distinguished by gate
        # (cold stays raw, all-missing stays truthful gated raw).
        if len(projections) == 1:
            by_expert: dict[int, list[str]] = {}
            for (_projection, expert), member in sorted(members.items()):
                by_expert.setdefault(int(expert), []).append(str(member))
            rows: list[torch.Tensor] = []
            for expert in sorted(by_expert):
                candidates = [self._direct(name) for name in by_expert[expert]]
                candidates = [value for value in candidates if value is not None]
                if not candidates:
                    # Preserve cold empty — do not raise; gate will mark ineligible raw
                    rows.append(torch.empty((0, 0), dtype=torch.float32))
                    continue
                reference = candidates[0]
                if any(
                    tuple(value.shape) != tuple(reference.shape)
                    or not torch.equal(value, reference)
                    for value in candidates[1:]
                ):
                    raise ValueError(
                        f"{qname}: fused expert {expert} members disagree on "
                        "their captured LDLQ input rows"
                    )
                rows.append(reference)
            return tuple(rows)
        # Multi-member fused (gate_up_proj): deterministic profile-order concat along rows.
        # Build per-expert mapping proj -> logical file qname.
        by_expert_map: dict[int, dict[str, str]] = {}
        for (proj, eid), logical_qname in members.items():
            by_expert_map.setdefault(int(eid), {})[str(proj)] = str(logical_qname)
        rows: list[torch.Tensor] = []
        for expert in sorted(by_expert_map):
            proj_map = by_expert_map[expert]
            # Verify all declared projections present in map for this expert
            missing_proj = [p for p in projections if p not in proj_map]
            if missing_proj:
                raise ValueError(
                    f"{qname}: fused expert {expert} missing declarations for projections {missing_proj}"
                )
            tensors: list[torch.Tensor | None] = []
            present_flags: list[bool] = []
            for proj in projections:
                qname_member = proj_map[proj]
                t = self._direct(qname_member)
                if t is not None:
                    if not isinstance(t, torch.Tensor) or t.ndim != 2:
                        raise ValueError(
                            f"{qname}: fused expert {expert} member {qname_member!r} has no rank-2 inputs"
                        )
                    tensors.append(t)
                    present_flags.append(True)
                else:
                    tensors.append(None)
                    present_flags.append(False)
            # All-or-nothing
            if any(present_flags) and not all(present_flags):
                present_projs = [p for p, f in zip(projections, present_flags) if f]
                missing_projs = [p for p, f in zip(projections, present_flags) if not f]
                raise ValueError(
                    f"{qname}: fused expert {expert} members partial presence — "
                    f"present {present_projs} missing {missing_projs} — all declared members must be present together or absent together"
                )
            if not any(present_flags):
                # Both absent -> empty placeholder for pooled prior
                rows.append(torch.empty((0, 0), dtype=torch.float32))
                continue
            # All present: validate same width and equal row count
            widths = {int(t.shape[1]) for t in tensors if t is not None}
            if len(widths) != 1:
                raise ValueError(
                    f"{qname}: fused expert {expert} members disagree on input width {widths} — projections {list(zip(projections, [tuple(t.shape) if t is not None else None for t in tensors]))}"
                )
            row_counts = {int(t.shape[0]) for t in tensors if t is not None}
            if len(row_counts) != 1:
                raise ValueError(
                    f"{qname}: fused expert {expert} members disagree on row count {row_counts} — simple concat requires equal counts"
                )
            # Deterministic profile-order concat along rows (dim 0)
            concatenated = torch.cat([t for t in tensors if t is not None], dim=0).contiguous()
            rows.append(concatenated)
        return tuple(rows)

    def load(
        self,
        qname: str,
        *,
        stack_size: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        direct = self._direct(qname)
        if direct is not None:
            return direct
        per_expert = self._per_expert(qname)
        if per_expert is not None:
            if stack_size is not None and len(per_expert) != int(stack_size):
                raise ValueError(
                    f"{qname}: LDLQ activation stacks={len(per_expert)} != "
                    f"weight stacks={stack_size}"
                )
            return per_expert

        from .moe_imatrix import (
            RoutedActivationSamples,
            synthesize_packed_expert_activation_samples,
        )

        replayed = synthesize_packed_expert_activation_samples(
            self.model_dir,
            self.activation_cache_dir,
            {str(qname)},
            self.profile,
            device=self.replay_device,
        ).get(str(qname))
        if isinstance(replayed, RoutedActivationSamples):
            if stack_size is None:
                return replayed.values
            # Replay fallback must return one original per-expert tensor with explicit
            # empty entries for missing experts — no pooling. Streaming export uses
            # this loader directly, so pooling here would factor and ship LDLQ for an
            # expert the cost path calls raw. Preserve cold empties for eligible-only gate.
            rows = tuple(
                replayed.values[replayed.expert_indices == expert].contiguous()
                for expert in range(int(stack_size))
            )
            return rows
        if isinstance(replayed, torch.Tensor) and replayed.ndim == 2:
            return replayed.detach().to(torch.float32).contiguous()
        raise ValueError(
            f"{qname}: no value-bearing activation rows for LDLQ export; "
            "rebuild the production activation cache"
        )


__all__ = [
    "CBLDLQActivationLoader",
    "NoObservedExpertRowsError",
    "fill_empty_expert_activation_rows",
]
