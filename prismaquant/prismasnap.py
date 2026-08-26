"""PrismaSnap: codec-selected, exactly invariant offline scale folds.

PrismaSnap is deliberately *not* a quantization format and does not alter the
PrismaQuant stage graph.  It prepares an ordinary floating-point Hugging Face
checkpoint.  The existing probe, allocator, production cache, recache,
export, and validation paths then consume that checkpoint unchanged.

The search in this module is pure tensor code.  Checkpoint discovery,
streaming materialization, receipts, and resume live in
``prismasnap_checkpoint`` so the numerical contract is independently unit
testable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch

from .export_native_compressed import (
    NVFP4_SCALE_RULE_STATIC_6,
    nvfp4_global_real,
    render_nvfp4_dequant,
    resolve_nvfp4_scale_rule,
)


PRISMASNAP_ALGORITHM = "prismasnap.diagonal_scale_fold.v1"
TransformMode = Literal["column_inverse", "row", "stationary"]
MaterializeMode = Literal["multiply", "divide", "affine_multiply"]


@dataclass(frozen=True)
class PrismaSnapSearchConfig:
    """Versioned search knobs; no behavior is read from the environment."""

    group_size: int = 16
    alphas: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5)
    max_rounds: int = 4
    stage: bool = True
    polish: bool = True
    polish_top: int = 8
    polish_pool: int = 16
    # The measured SnapQuant A/B selected its source transform against plain
    # static-6 NVFP4, then let the ordinary downstream pipeline apply JSO.
    # Keep that treatment contract unchanged for the first production gate.
    scale_rule: str = NVFP4_SCALE_RULE_STATIC_6
    snapped_scale_scoring: bool = False
    joint_scale_levels: tuple[float, ...] = (6.0, 4.0)

    def __post_init__(self) -> None:
        if self.group_size != 16:
            raise ValueError("PrismaSnap production search requires group_size=16")
        if not self.alphas or self.alphas[0] != 0.0:
            raise ValueError("PrismaSnap alphas must begin with the no-op 0.0")
        if any(not math.isfinite(a) or a < 0.0 for a in self.alphas):
            raise ValueError("PrismaSnap alphas must be finite and non-negative")
        if len(set(self.alphas)) != len(self.alphas):
            raise ValueError("PrismaSnap alphas must be unique")
        if self.max_rounds < 1 or self.max_rounds > 16:
            raise ValueError("PrismaSnap max_rounds must be in [1, 16]")
        if self.polish_top < 0 or self.polish_pool < 0:
            raise ValueError("PrismaSnap polish limits must be non-negative")
        if self.polish_top > self.polish_pool:
            raise ValueError("PrismaSnap polish_top cannot exceed polish_pool")
        if (
            not self.joint_scale_levels
            or len(set(self.joint_scale_levels)) != len(self.joint_scale_levels)
            or any(
                not math.isfinite(level) or level <= 0.0
                for level in self.joint_scale_levels
            )
        ):
            raise ValueError(
                "PrismaSnap joint_scale_levels must be unique, finite, positive"
            )
        object.__setattr__(
            self, "scale_rule", resolve_nvfp4_scale_rule(self.scale_rule)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": PRISMASNAP_ALGORITHM,
            "group_size": self.group_size,
            "alphas": list(self.alphas),
            "max_rounds": self.max_rounds,
            "variant": [
                name
                for name, enabled in (("stage", self.stage), ("polish", self.polish))
                if enabled
            ],
            "polish_top": self.polish_top,
            "polish_pool": self.polish_pool,
            "nvfp4_scale_rule": self.scale_rule,
            "nvfp4_snapped_scale_scoring": self.snapped_scale_scoring,
            "nvfp4_joint_scale_levels": list(self.joint_scale_levels),
            # These are value-bearing algorithm semantics, not commentary.
            # They reproduce the stage,polish implementation that generated
            # the release evidence in snapquant_trial.py.
            "objective_fold_dtype": "float32",
            "global_scale_scope": "per_tensor",
            "materialization_rounding": "sequential_bf16_per_transform",
        }


@dataclass
class PrismaSnapConsumer:
    """One exact-codec objective term attached to an invariance seam.

    The production-v1 search derives one global scale per logical tensor.  It
    intentionally reproduces the measured ``stage,polish`` prototype rather
    than introducing fused-sibling scoring into the release arm.
    """

    name: str
    weight: torch.Tensor
    importance: torch.Tensor
    mode: TransformMode
    # Research extensions may need multiple physical source tensors to share
    # one production codec global (for example per-expert gate/up Linears that
    # vLLM packs into one logical gate_up tensor). Dense measured-v1 leaves
    # this unset, preserving its exact per-source-tensor semantics.
    codec_group_name: str | None = None

    def validate(self, n_channels: int, group_size: int) -> None:
        if not self.name:
            raise ValueError("PrismaSnap consumer name cannot be empty")
        if self.codec_group_name is not None and not self.codec_group_name:
            raise ValueError("PrismaSnap codec group override cannot be empty")
        if self.weight.ndim != 2:
            raise ValueError(
                f"{self.name}: PrismaSnap scalar codec requires rank-2 weight, "
                f"got {tuple(self.weight.shape)}"
            )
        rows, cols = map(int, self.weight.shape)
        if cols % group_size:
            raise ValueError(
                f"{self.name}: NVFP4 group_size={group_size} does not divide {cols}"
            )
        if self.mode == "column_inverse" and cols != n_channels:
            raise ValueError(
                f"{self.name}: column seam has {cols} channels, expected {n_channels}"
            )
        if self.mode == "row" and rows != n_channels:
            raise ValueError(
                f"{self.name}: row seam has {rows} channels, expected {n_channels}"
            )
        if self.mode not in {"column_inverse", "row", "stationary"}:
            raise ValueError(f"{self.name}: unsupported PrismaSnap mode {self.mode!r}")
        imp = self.importance
        if imp.ndim != 1 or int(imp.numel()) != cols:
            raise ValueError(
                f"{self.name}: importance shape {tuple(imp.shape)} does not match "
                f"weight input width {cols}"
            )
        if not bool(torch.isfinite(self.weight).all().item()):
            raise ValueError(f"{self.name}: weight contains non-finite values")
        if not bool(torch.isfinite(imp).all().item()):
            raise ValueError(f"{self.name}: importance contains non-finite values")
        if bool((imp < 0).any().item()):
            raise ValueError(f"{self.name}: importance contains negative values")
        if not bool((imp > 0).any().item()):
            raise ValueError(f"{self.name}: importance has no positive mass")

    @property
    def codec_group(self) -> str:
        return self.codec_group_name or self.name


def _fold_weight(
    consumer: PrismaSnapConsumer,
    scale: torch.Tensor,
) -> torch.Tensor:
    weight = consumer.weight.to(torch.float64)
    if consumer.mode == "column_inverse":
        folded = weight / scale.view(1, -1)
    elif consumer.mode == "row":
        folded = weight * scale.view(-1, 1)
    else:
        folded = weight
    # Exact reproduction of the measured prototype: source BF16 parameters
    # were promoted to fp32 when captured, and candidate folds were performed
    # in fp64 then rounded once to fp32 for the codec objective.
    return folded.to(torch.float32)


def _importance_for(
    consumer: PrismaSnapConsumer,
    scale: torch.Tensor,
) -> torch.Tensor:
    importance = consumer.importance.to(
        device=consumer.weight.device, dtype=torch.float64
    )
    if consumer.mode == "column_inverse":
        importance = importance * scale.square()
    return importance.to(torch.float32).clamp_min_(1e-12)


def _shared_globals(
    consumers: Sequence[PrismaSnapConsumer],
    scale: torch.Tensor,
    config: PrismaSnapSearchConfig,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for consumer in consumers:
        folded = _fold_weight(consumer, scale)
        natural = nvfp4_global_real(
            folded,
            group_size=config.group_size,
            scale_rule=config.scale_rule,
            snapped_scale_scoring=config.snapped_scale_scoring,
            joint_scale_levels=config.joint_scale_levels,
        )
        key = consumer.codec_group
        prior = result.get(key)
        result[key] = natural if prior is None else torch.maximum(prior, natural)
    return result


def measured_render_objective(
    consumers: Sequence[PrismaSnapConsumer],
    scale: torch.Tensor,
    config: PrismaSnapSearchConfig,
) -> float:
    """Weighted SSE on the measured-v1 per-tensor production NVFP4 render."""
    globals_by_group = _shared_globals(consumers, scale, config)
    total = torch.zeros((), device=scale.device, dtype=torch.float64)
    for consumer in consumers:
        folded = _fold_weight(consumer, scale)
        rendered = render_nvfp4_dequant(
            folded,
            group_size=config.group_size,
            global_real_override=globals_by_group[consumer.codec_group],
            scale_rule=config.scale_rule,
            snapped_scale_scoring=config.snapped_scale_scoring,
            joint_scale_levels=config.joint_scale_levels,
        )
        error = (rendered - folded).square()
        error = error * _importance_for(consumer, scale).view(1, -1)
        total = total + error.to(torch.float64).sum()
    value = float(total.item())
    if not math.isfinite(value):
        raise RuntimeError("PrismaSnap measured render objective became non-finite")
    return value


def _cell_errors(
    consumer: PrismaSnapConsumer,
    scale: torch.Tensor,
    *,
    fixed_global: torch.Tensor,
    n_channels: int,
    config: PrismaSnapSearchConfig,
) -> torch.Tensor:
    """Return one frozen-global error value per seam group of 16."""
    n_groups = n_channels // config.group_size
    if consumer.mode == "stationary":
        # Its value is candidate-independent while the global is frozen.
        return torch.zeros(n_groups, device=scale.device, dtype=torch.float64)
    folded = _fold_weight(consumer, scale)
    rendered = render_nvfp4_dequant(
        folded,
        group_size=config.group_size,
        global_real_override=fixed_global,
        scale_rule=config.scale_rule,
        snapped_scale_scoring=config.snapped_scale_scoring,
        joint_scale_levels=config.joint_scale_levels,
    )
    error = (rendered - folded).square()
    error = error * _importance_for(consumer, scale).view(1, -1)
    rows, cols = map(int, folded.shape)
    if consumer.mode == "column_inverse":
        return error.reshape(
            rows, cols // config.group_size, config.group_size
        ).sum(dim=(0, 2), dtype=torch.float64)
    return error.reshape(
        rows // config.group_size, config.group_size, cols
    ).sum(dim=(1, 2), dtype=torch.float64)


def search_diagonal_scale(
    consumers: Sequence[PrismaSnapConsumer],
    reference_importance: torch.Tensor,
    *,
    config: PrismaSnapSearchConfig | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Run the production ``stage,polish`` per-group diagonal search.

    The fast phase evaluates all groups in batched codec passes under each
    logical tensor's current global, iterating globals to a fixed point.  A
    true full-render acceptance gate makes the no-op a hard upper bound.
    Targeted true-render polish then revisits at most 16 high-impact/global-
    setting groups, matching the production recipe measured in the handover.
    """
    cfg = config or PrismaSnapSearchConfig()
    if not consumers:
        raise ValueError("PrismaSnap seam has no consumers")
    if reference_importance.ndim != 1:
        raise ValueError("PrismaSnap reference importance must be rank 1")
    n_channels = int(reference_importance.numel())
    if n_channels == 0 or n_channels % cfg.group_size:
        raise ValueError(
            f"PrismaSnap seam width {n_channels} is not divisible by "
            f"group_size={cfg.group_size}"
        )
    device = consumers[0].weight.device
    if any(c.weight.device != device for c in consumers):
        raise ValueError("all PrismaSnap consumers for a seam must share a device")
    ref = reference_importance.to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(ref).all().item()) or bool((ref < 0).any().item()):
        raise ValueError("PrismaSnap reference importance must be finite/non-negative")
    if not bool((ref > 0).any().item()):
        raise ValueError("PrismaSnap reference importance has no positive mass")
    for consumer in consumers:
        consumer.validate(n_channels, cfg.group_size)

    ref_grouped = ref.clamp_min(1e-12).view(-1, cfg.group_size)
    geometric_mean = ref_grouped.log().mean(dim=1, keepdim=True).exp()
    relative = (ref_grouped / geometric_mean).reshape(-1)
    candidates = [relative.pow(-float(alpha)) for alpha in cfg.alphas]
    n_groups = n_channels // cfg.group_size

    scale = torch.ones(n_channels, device=device, dtype=torch.float64)
    last_scores: torch.Tensor | None = None
    rounds = 0
    for round_index in range(cfg.max_rounds):
        rounds = round_index + 1
        fixed_globals = _shared_globals(consumers, scale, cfg)
        per_candidate: list[torch.Tensor] = []
        for candidate in candidates:
            score = torch.zeros(n_groups, device=device, dtype=torch.float64)
            for consumer in consumers:
                score += _cell_errors(
                    consumer,
                    candidate,
                    fixed_global=fixed_globals[consumer.codec_group],
                    n_channels=n_channels,
                    config=cfg,
                )
            per_candidate.append(score)
        last_scores = torch.stack(per_candidate)
        best = last_scores.argmin(dim=0)
        if cfg.stage and round_index == 0:
            chosen = last_scores.gather(0, best.view(1, -1)).squeeze(0)
            gains = last_scores[0] - chosen
            best = torch.where(
                gains >= gains.median(), best, torch.zeros_like(best)
            )
        next_scale = torch.ones_like(scale)
        for group_index in range(n_groups):
            candidate_index = int(best[group_index].item())
            if candidate_index:
                start = group_index * cfg.group_size
                stop = start + cfg.group_size
                next_scale[start:stop] = candidates[candidate_index][start:stop]
        if torch.equal(next_scale, scale):
            break
        scale = next_scale

    assert last_scores is not None
    identity = torch.ones_like(scale)
    baseline = measured_render_objective(consumers, identity, cfg)
    current = measured_render_objective(consumers, scale, cfg)
    fell_back = False
    if current >= baseline:
        fell_back = True
        gains = last_scores[0] - last_scores.min(dim=0).values
        keep = gains >= gains.median()
        reduced = torch.ones_like(scale)
        mask = keep.repeat_interleave(cfg.group_size)
        reduced[mask] = scale[mask]
        reduced_objective = measured_render_objective(consumers, reduced, cfg)
        if reduced_objective < baseline:
            scale, current = reduced, reduced_objective
        else:
            scale, current = identity, baseline

    polished = 0
    polish_count = 0
    if cfg.polish and cfg.polish_pool:
        gains = last_scores[0] - last_scores.min(dim=0).values
        pool = set(
            torch.argsort(gains, descending=True)[: cfg.polish_top].tolist()
        )
        for consumer in consumers:
            if consumer.mode == "stationary":
                continue
            folded_abs = _fold_weight(consumer, scale).abs()
            threshold = folded_abs.max() * 0.5
            if consumer.mode == "column_inverse":
                hit = (
                    (folded_abs >= threshold)
                    .any(dim=0)
                    .view(-1, cfg.group_size)
                    .any(dim=1)
                )
            else:
                hit = (
                    (folded_abs >= threshold)
                    .any(dim=1)
                    .view(-1, cfg.group_size)
                    .any(dim=1)
                )
            pool.update(torch.nonzero(hit).flatten().tolist())
        order = sorted(pool, key=lambda index: -float(gains[index]))[
            : cfg.polish_pool
        ]
        polish_count = len(order)
        for group_index in order:
            start = group_index * cfg.group_size
            stop = start + cfg.group_size
            best_scale: torch.Tensor | None = None
            best_objective = current
            for candidate in candidates:
                proposal = scale.clone()
                proposal[start:stop] = candidate[start:stop]
                if torch.equal(proposal, scale):
                    continue
                objective = measured_render_objective(consumers, proposal, cfg)
                if objective < best_objective:
                    best_scale, best_objective = proposal, objective
            if best_scale is not None:
                scale, current = best_scale, best_objective
                polished += 1

    moved = int(
        (scale.view(-1, cfg.group_size) != 1.0).any(dim=1).sum().item()
    )
    stats: dict[str, object] = {
        "algorithm": PRISMASNAP_ALGORITHM,
        "error_baseline": baseline,
        "error_final": current,
        "improvement_fraction": (
            0.0 if baseline == 0.0 else (baseline - current) / baseline
        ),
        "groups": n_groups,
        "groups_moved": moved,
        "rounds": rounds,
        "candidate_count": len(candidates),
        "fell_back": fell_back,
        "polish_pool": polish_count,
        "polished": polished,
        "variant": [
            name
            for name, enabled in (("stage", cfg.stage), ("polish", cfg.polish))
            if enabled
        ],
    }
    return scale.detach(), stats


def apply_diagonal_transform(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    mode: MaterializeMode,
    axis: int,
    *,
    parameter_offset: float = 0.0,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply one plan transform in fp64 and restore the source dtype."""
    if tensor.ndim not in {1, 2, 3}:
        raise ValueError(f"unsupported PrismaSnap tensor rank {tensor.ndim}")
    if axis < 0:
        axis += tensor.ndim
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"invalid PrismaSnap transform axis {axis}")
    if int(tensor.shape[axis]) != int(scale.numel()):
        raise ValueError(
            f"PrismaSnap transform width {scale.numel()} does not match "
            f"axis-{axis} size {tensor.shape[axis]}"
        )
    shape = [1] * tensor.ndim
    shape[axis] = int(scale.numel())
    factor = scale.to(device=tensor.device, dtype=torch.float64).reshape(shape)
    value = tensor.to(torch.float64)
    if not math.isfinite(parameter_offset):
        raise ValueError("PrismaSnap transform parameter offset must be finite")
    if mode == "multiply":
        if parameter_offset != 0.0:
            raise ValueError("raw multiply cannot carry a parameter offset")
        value = value * factor
    elif mode == "divide":
        if parameter_offset != 0.0:
            raise ValueError("divide cannot carry a parameter offset")
        value = value / factor
    elif mode == "affine_multiply":
        # Some RMSNorm families store a zero-centered parameter while the
        # executed multiplier is (parameter + offset).  Fold the effective
        # gamma, then map back into the checkpoint's parameter coordinates.
        value = (value + parameter_offset) * factor - parameter_offset
    else:
        raise ValueError(f"unsupported PrismaSnap transform operation {mode!r}")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError("PrismaSnap transform produced non-finite values")
    result = value.to(dtype=output_dtype or tensor.dtype).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError(
            "PrismaSnap transform overflowed while restoring the checkpoint dtype"
        )
    return result
