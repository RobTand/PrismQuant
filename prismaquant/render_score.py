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


def normalize_row_weights(
    row_weights: torch.Tensor | None,
    n_rows: int,
    device: torch.device,
    *,
    clip_env: str = "PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP",
    default_clip: float = 64.0,
) -> torch.Tensor | None:
    """Return nonnegative row weights normalized to mean 1."""

    if row_weights is None or n_rows <= 0:
        return None
    try:
        rw = row_weights.detach().reshape(-1).to(device=device, dtype=torch.float32)
    except Exception:
        return None
    if rw.numel() < n_rows:
        return None
    rw = torch.nan_to_num(rw[:n_rows], nan=0.0, posinf=0.0, neginf=0.0)
    rw = rw.clamp_min(0.0)
    rw = rw / rw.mean().clamp_min(1e-12)
    try:
        clip = float(os.environ.get(clip_env, str(default_clip)))
    except Exception:
        clip = float(default_clip)
    if clip > 0.0:
        rw = rw.clamp_max(float(clip))
        rw = rw / rw.mean().clamp_min(1e-12)
    return rw


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
    # Trellis TCQ: 256-state tail-biting Viterbi over a per-input-column
    # imatrix-weighted objective. The wire bytes (not the decoded float
    # tensor) are the shipped artifact, so the render, surrogate pricing,
    # KL validation and export must be byte-identical (principle 8). This
    # mechanism is the deliberate trellis render, not a candidate layered on
    # RTN, exactly like weighted_vq for CB/GGUF but with a distinct kernel
    # family (tcq_trellis). It is also imatrix-weighted, so it participates
    # in the same WEIGHTED_RENDER_FAMILIES gate as CB/GGUF.
    register_render_mechanism(RenderMechanismSpec(
        name="trellis",
        operation="imatrix_weighted_trellis_search",
        scope="linear",
        phase=50,
        gate_metric="weight_mse",
        description=(
            "Gridbook trellis-coded quantization (TCQ) Viterbi search — "
            "the 256-state tail-biting encoder whose wire bytes ARE the "
            "shipped artifact. The decode of those exact bytes is the "
            "returned tensor, so render/surrogate/KL/export share one "
            "encode (one-encode invariant, WO-B)."
        ),
    ))


_register_builtins()
