"""Shared local render scoring and mechanism ordering.

The scorer is the common acceptance metric for local per-Linear or
fused-sibling transforms: compare the candidate rendered weight against the
current baseline on the same activation rows, optionally weighted by per-token
Fisher/output importance. Lower is better.

Global basis transforms are intentionally outside this local gate; they must
be evaluated as full recipe arms.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
import os
from typing import Iterable, Mapping

import torch


@dataclass(frozen=True)
class RenderGateDecision:
    accepted: bool
    baseline_score: float
    candidate_score: float
    relative_gain: float
    metric: str
    reason: str


# Fisher row-weight clip: single source of truth (issue #159).
#
# The render scorer, the quant-cost probe, and the native exporter all weight
# activation rows by per-token Fisher (gradient-squared) importance normalized
# to mean 1, then cap the largest weight so one hot token cannot dominate the
# local objective. The cap lives here, and only here: every other module
# resolves it through resolve_fisher_row_weight_clip() and applies it through
# normalize_clipped_fisher_row_weights(). A second copy of the
# normalise-then-clip arithmetic anywhere else is a drift bug by definition.
FISHER_ROW_WEIGHT_CLIP_DEFAULT = 64.0
FISHER_ROW_WEIGHT_CLIP_ENV = "PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP"
# Older spelling from the GPTQ-named era, kept as a fallback so existing
# scripts and reproductions keep working. New code sets the canonical name.
FISHER_ROW_WEIGHT_CLIP_ALIAS_ENV = "PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP"

# Derivation gap (house principle 2: thresholds come from the objective, not
# intuition): 64 has no in-repo derivation -- no sweep, KL/bpp comparison, or
# analysis selects it. What is known is the computable bound: a mean-1 vector
# over n_rows rows reaches at most n_rows (a fully concentrated row is exactly
# n_rows), so any cap below n_rows binds by construction, and 64 is below
# n_rows for every production calibration (n_rows is in the thousands). The
# cap is therefore on the binding side everywhere it matters, by arithmetic
# rather than by measurement. Whether 64 is the right value is a separate
# measured question; do not retune it here.


def resolve_fisher_row_weight_clip() -> float:
    """Return the Fisher row-weight clip from the environment.

    The canonical var wins; the alias is a documented fallback; anything
    unreadable falls back to the default.
    """
    try:
        return float(os.environ.get(
            FISHER_ROW_WEIGHT_CLIP_ENV,
            os.environ.get(
                FISHER_ROW_WEIGHT_CLIP_ALIAS_ENV,
                str(FISHER_ROW_WEIGHT_CLIP_DEFAULT),
            ),
        ))
    except Exception:
        return float(FISHER_ROW_WEIGHT_CLIP_DEFAULT)


def normalize_clipped_fisher_row_weights(
    rw: torch.Tensor,
    clip: float,
    *,
    require_positive_mean: bool,
) -> torch.Tensor | None:
    """Mean-1 normalize ``rw``, clamp it at ``clip``, renormalize to mean 1.

    This is the one copy of the normalise-then-clip arithmetic shared by the
    render scorer, the quant-cost probe, and the native exporter. ``rw`` is
    the already-sliced 1-D float32 vector on its target device. When
    ``require_positive_mean`` is set, a degenerate (all-zero or non-finite
    mean) input returns None so the caller falls back to the unweighted
    objective; otherwise the legacy scorer policy applies and the degenerate
    input comes back as zeros.
    """
    rw = torch.nan_to_num(rw, nan=0.0, posinf=0.0, neginf=0.0)
    rw = rw.clamp_min(0.0)
    mean = rw.mean()
    if require_positive_mean and (
        not torch.isfinite(mean) or float(mean.item()) <= 0.0
    ):
        return None
    rw = rw / mean.clamp_min(1e-12)
    if clip > 0.0:
        rw = rw.clamp_max(float(clip))
        mean2 = rw.mean()
        if torch.isfinite(mean2) and float(mean2.item()) > 0.0:
            rw = rw / mean2.clamp_min(1e-12)
    return rw


def normalize_row_weights(
    row_weights: torch.Tensor | None,
    n_rows: int,
    device: torch.device,
    *,
    clip_env: str = FISHER_ROW_WEIGHT_CLIP_ALIAS_ENV,
    default_clip: float = FISHER_ROW_WEIGHT_CLIP_DEFAULT,
) -> torch.Tensor | None:
    """Return nonnegative row weights normalized to mean 1.

    With default arguments the clip is the single-sourced
    resolve_fisher_row_weight_clip(). ``clip_env``/``default_clip`` are a
    deprecated explicit override kept so older call sites keep working; no
    in-repo caller passes them.
    """

    if row_weights is None or n_rows <= 0:
        return None
    try:
        rw = row_weights.detach().reshape(-1).to(device=device, dtype=torch.float32)
    except Exception:
        return None
    if rw.numel() < n_rows:
        return None
    rw = rw[:n_rows]
    if (
        clip_env != FISHER_ROW_WEIGHT_CLIP_ALIAS_ENV
        or default_clip != FISHER_ROW_WEIGHT_CLIP_DEFAULT
    ):
        try:
            clip = float(os.environ.get(clip_env, str(default_clip)))
        except Exception:
            clip = float(default_clip)
    else:
        clip = resolve_fisher_row_weight_clip()
    return normalize_clipped_fisher_row_weights(
        rw, clip, require_positive_mean=False
    )


def score_render_error(
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    row_weights: torch.Tensor | None = None,
    row_chunk: int = 128,
) -> float:
    """Mean output-space MSE for one rendered weight candidate.

    With ``row_weights`` this becomes the Fisher/output-weighted local
    objective.
    """

    rows, cols = reference_weight.shape
    if rendered_weight.shape != reference_weight.shape:
        return float("inf")
    if activations.shape[-1] != cols:
        return float("inf")
    device = reference_weight.device
    diff_t = (
        reference_weight.detach().to(device=device, dtype=torch.float32)
        - rendered_weight.detach().to(device=device, dtype=torch.float32)
    ).t().contiguous()
    x = activations.detach().to(device=device, dtype=torch.float32).reshape(-1, cols)
    rw = normalize_row_weights(row_weights, x.shape[0], device)
    total = torch.zeros((), dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, x.shape[0], int(row_chunk)):
            y = x[start:start + int(row_chunk)] @ diff_t
            err = y.pow(2)
            if rw is not None:
                err = err * rw[start:start + err.shape[0]].unsqueeze(1)
            total = total + err.sum()
    return float(total.item()) / max(1, int(x.shape[0]) * int(rows))


def gate_render_candidate(
    *,
    baseline_score: float,
    candidate_score: float,
    metric: str,
    min_relative_gain: float = 0.0,
) -> RenderGateDecision:
    """Accept a candidate only when it improves the active score."""

    denom = max(abs(float(baseline_score)), 1e-30)
    relative_gain = (float(baseline_score) - float(candidate_score)) / denom
    accepted = (
        math.isfinite(float(candidate_score))
        and float(candidate_score) < float(baseline_score)
        and float(relative_gain) >= float(min_relative_gain)
    )
    if accepted:
        reason = "improved"
    elif float(candidate_score) >= float(baseline_score):
        reason = "regressed_or_tied"
    else:
        reason = "below_min_gain"
    return RenderGateDecision(
        accepted=bool(accepted),
        baseline_score=float(baseline_score),
        candidate_score=float(candidate_score),
        relative_gain=float(relative_gain if accepted else 0.0),
        metric=str(metric),
        reason=reason,
    )


def output_error_distribution_stats(
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    row_chunk: int = 128,
) -> dict[str, float]:
    rows, cols = reference_weight.shape
    if (
        rendered_weight.shape != reference_weight.shape
        or activations.shape[-1] != cols
    ):
        return {
            "mse": float("inf"),
            "p50_abs": float("inf"),
            "p99_abs": float("inf"),
            "max_abs": float("inf"),
            "tail_ratio_p99_p50": float("inf"),
        }
    device = reference_weight.device
    diff_t = (
        reference_weight.detach().to(device=device, dtype=torch.float32)
        - rendered_weight.detach().to(device=device, dtype=torch.float32)
    ).t().contiguous()
    x = activations.detach().to(device=device, dtype=torch.float32).reshape(-1, cols)
    chunks: list[torch.Tensor] = []
    total_sq = torch.zeros((), dtype=torch.float32, device=device)
    n_values = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], int(row_chunk)):
            y_abs = (x[start:start + int(row_chunk)] @ diff_t).abs()
            total_sq = total_sq + y_abs.pow(2).sum()
            n_values += int(y_abs.numel())
            chunks.append(y_abs.detach().flatten())
    if n_values <= 0:
        return {
            "mse": 0.0,
            "p50_abs": 0.0,
            "p99_abs": 0.0,
            "max_abs": 0.0,
            "tail_ratio_p99_p50": 0.0,
        }
    values = torch.cat(chunks).to(torch.float32)
    p50 = float(values.quantile(0.50).item())
    p99 = float(values.quantile(0.99).item())
    max_abs = float(values.max().item())
    return {
        "mse": float(total_sq.item()) / n_values,
        "p50_abs": p50,
        "p99_abs": p99,
        "max_abs": max_abs,
        "tail_ratio_p99_p50": float(p99 / max(p50, 1e-12)),
    }


@dataclass(frozen=True)
class RenderMechanismSpec:
    name: str
    operation: str
    scope: str
    phase: int
    gate_metric: str
    after: tuple[str, ...] = ()
    before: tuple[str, ...] = ()
    exclusive_group: str | None = None
    description: str = ""


@dataclass(frozen=True)
class RenderMechanismPlan:
    ordered: tuple[RenderMechanismSpec, ...]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.ordered)


_REGISTRY: dict[str, RenderMechanismSpec] = {}


def register_render_mechanism(spec: RenderMechanismSpec) -> None:
    if not spec.name:
        raise ValueError("render mechanism name must be non-empty")
    _REGISTRY[spec.name] = spec


def render_mechanism(name: str) -> RenderMechanismSpec:
    return _REGISTRY[str(name)]


def registered_render_mechanisms() -> Mapping[str, RenderMechanismSpec]:
    return dict(_REGISTRY)


def resolve_render_mechanism_order(
    enabled: Iterable[str],
    *,
    strict_unknown: bool = True,
) -> RenderMechanismPlan:
    """Order mechanisms from declared operation semantics.

    Plugins declare a phase plus optional before/after constraints. This keeps
    ordering tied to what a transform *does* rather than the spelling order in
    an env var.
    """

    requested = tuple(
        dict.fromkeys(
            str(name).strip()
            for name in enabled
            if str(name).strip()
        )
    )
    warnings: list[str] = []
    errors: list[str] = []
    specs: dict[str, RenderMechanismSpec] = {}
    for name in requested:
        spec = _REGISTRY.get(name)
        if spec is None:
            msg = f"unknown render mechanism: {name}"
            if strict_unknown:
                errors.append(msg)
            else:
                warnings.append(msg)
            continue
        specs[name] = spec

    exclusive_seen: dict[str, str] = {}
    for spec in specs.values():
        if spec.exclusive_group is None:
            continue
        other = exclusive_seen.get(spec.exclusive_group)
        if other is not None:
            errors.append(
                f"{other} and {spec.name} are mutually exclusive "
                f"({spec.exclusive_group})"
            )
        exclusive_seen[spec.exclusive_group] = spec.name

    edges: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = {name: 0 for name in specs}
    for a_name, a in specs.items():
        for b_name, b in specs.items():
            if a_name == b_name:
                continue
            if a.phase < b.phase:
                edges[a_name].add(b_name)
            elif a.phase > b.phase:
                edges[b_name].add(a_name)
    for name, spec in specs.items():
        for dep in spec.after:
            if dep in specs:
                edges[dep].add(name)
        for nxt in spec.before:
            if nxt in specs:
                edges[name].add(nxt)

    for src, dsts in edges.items():
        for dst in dsts:
            if src == dst:
                continue
            indegree[dst] += 1

    ready = deque(sorted(
        (name for name, deg in indegree.items() if deg == 0),
        key=lambda n: (specs[n].phase, n),
    ))
    ordered_names: list[str] = []
    while ready:
        name = ready.popleft()
        ordered_names.append(name)
        for dst in sorted(edges.get(name, ())):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
        ready = deque(sorted(ready, key=lambda n: (specs[n].phase, n)))

    if len(ordered_names) != len(specs):
        errors.append("render mechanism ordering has a dependency cycle")
        ordered_names = sorted(specs, key=lambda n: (specs[n].phase, n))

    return RenderMechanismPlan(
        ordered=tuple(specs[name] for name in ordered_names),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _register_builtins() -> None:
    register_render_mechanism(RenderMechanismSpec(
        name="four_over_six",
        operation="nvfp4_scale_rule",
        scope="nvfp4_block",
        phase=40,
        gate_metric="output_mse",
        description="NVFP4 per-block scale rule candidate; uses the same runtime kernel.",
    ))
    register_render_mechanism(RenderMechanismSpec(
        name="gptq",
        operation="rounding_solver",
        scope="linear",
        phase=50,
        gate_metric="output_mse",
        after=("four_over_six",),
        description="Local OBS/GPTQ rounding.",
    ))
    register_render_mechanism(RenderMechanismSpec(
        name="static_act_order",
        operation="rounding_solver_modifier",
        scope="linear",
        phase=50,
        gate_metric="output_mse",
        before=("gptq",),
        description="Static activation-order GPTQ without runtime permutation.",
    ))
    register_render_mechanism(RenderMechanismSpec(
        name="joint_scale_opt",
        operation="nvfp4_scale_optimizer",
        scope="linear",
        phase=50,
        gate_metric="output_mse",
        after=("four_over_six",),
        before=("gptq",),
        description="Joint NVFP4 tensor/global scale search inside GPTQ.",
    ))
    register_render_mechanism(RenderMechanismSpec(
        name="fisher_gptq",
        operation="rounding_solver_modifier",
        scope="linear",
        phase=50,
        gate_metric="fisher_output_mse",
        before=("gptq",),
        description="Fisher/output-weighted GPTQ objective.",
    ))
    register_render_mechanism(RenderMechanismSpec(
        name="weighted_vq",
        operation="imatrix_weighted_search",
        scope="linear",
        phase=50,
        gate_metric="weight_mse",
        description=(
            "Imatrix (per-input-column) weighted search for the CB codebook "
            "and GGUF k-quant families — their exporters always render "
            "weighted, so this IS their deliberate render rather than a "
            "candidate layered on RTN (re-vet R3 / CB Milestone C)."
        ),
    ))
    register_render_mechanism(RenderMechanismSpec(
        name="scale_sweep",
        operation="codebook_scale_refine",
        scope="linear",
        phase=60,
        gate_metric="output_mse",
        after=("gptq", "fisher_gptq"),
        description="Closed-form NVFP4/MXFP8_E4M3 scale refinement.",
    ))


_register_builtins()
