"""Research-only MoE algebra and codec objectives for PrismaSnap.

This module deliberately does not promote MoE checkpoints into the dense
PrismaSnap production contract.  It supplies the numerical building blocks
used by the schema-separated MoE planner in :mod:`prismasnap_checkpoint`:

* one post-norm scale shared by every direct consumer, while router logits
  are compensated exactly but excluded from the codec objective;
* one independent up-row/down-column scale per routed expert; and
* production-faithful NVFP4 scoring of packed ``[E, M, N]`` weights, with a
  separate global for each expert's logical serving tensor.

The dense measured-v1 implementation remains in :mod:`prismasnap` and is not
parameterized through this research lane.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch

from .prismasnap import (
    PrismaSnapConsumer,
    PrismaSnapSearchConfig,
    search_diagonal_scale,
)


PRISMASNAP_MOE_ALGORITHM = "prismasnap.diagonal_scale_fold.moe_research.v1"
PRISMASNAP_MOE_PROMOTION = "RESEARCH_ONLY_REQUIRES_REAL_MOE_FOLD_KL"


def moe_search_contract(config: PrismaSnapSearchConfig) -> dict[str, object]:
    """Return the closed, schema-bearing MoE search treatment.

    The underlying scalar search knobs intentionally match measured dense v1,
    but this is a different algorithm identity because its objective has an
    expert axis and its materialization program contains expert slices.
    """
    dense = config.as_dict()
    dense["algorithm"] = PRISMASNAP_MOE_ALGORITHM
    dense.update(
        {
            "expert_global_scope": "per_expert_logical_tensor",
            "routed_gate_up_global_scope": "per_expert_joint_gate_up",
            "shared_gate_up_global_scope": "per_shared_expert_joint_gate_up",
            "down_global_scope": "per_expert_independent",
            "packed_expert_axis": 0,
            "router_codec_objective": "excluded_exact_compensation_only",
            "expert_seam_scope": "independent_per_expert",
            "expert_coverage_policy": "all_experts_routed",
            "promotion": PRISMASNAP_MOE_PROMOTION,
        }
    )
    return dense


def _positive_importance(
    value: torch.Tensor,
    *,
    expected: tuple[int, ...],
    where: str,
) -> torch.Tensor:
    if tuple(value.shape) != expected:
        raise ValueError(
            f"{where} importance shape {tuple(value.shape)} != {expected}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{where} importance contains non-finite values")
    if bool((value < 0).any().item()):
        raise ValueError(f"{where} importance contains negative values")
    if not bool((value.sum(dim=-1) > 0).all().item()):
        raise ValueError(f"{where} has an expert with no positive importance mass")
    return value


@dataclass(frozen=True)
class PackedGateUp:
    """One routed packed gate/up tensor and its per-expert input moments.

    ``weight`` is ``[E, 2I, H]``.  The profile declares the gate/up row split;
    the numerical core never infers it from a Qwen name.  ``importance`` is
    ``[E, H]`` from the existing packed-expert probe contract.
    """

    name: str
    weight: torch.Tensor
    importance: torch.Tensor
    gate_rows: tuple[int, int]
    up_rows: tuple[int, int]

    def validate(self, *, group_size: int) -> tuple[int, int, int]:
        if not self.name:
            raise ValueError("packed gate/up name cannot be empty")
        if self.weight.ndim != 3:
            raise ValueError(
                f"{self.name}: packed gate/up must be rank 3, got "
                f"{tuple(self.weight.shape)}"
            )
        experts, rows, hidden = map(int, self.weight.shape)
        if experts <= 0 or rows <= 0 or hidden <= 0:
            raise ValueError(f"{self.name}: packed gate/up has an empty axis")
        if hidden % group_size:
            raise ValueError(
                f"{self.name}: hidden width {hidden} is not divisible by "
                f"group_size={group_size}"
            )
        gate_start, gate_stop = self.gate_rows
        up_start, up_stop = self.up_rows
        if not (
            0 <= gate_start < gate_stop <= rows
            and 0 <= up_start < up_stop <= rows
            and gate_stop - gate_start == up_stop - up_start
            and {gate_start, up_start} == {0, gate_stop}
            and max(gate_stop, up_stop) == rows
        ):
            raise ValueError(
                f"{self.name}: gate/up rows must be two equal, complete, "
                f"non-overlapping contiguous slices; got "
                f"gate={self.gate_rows} up={self.up_rows} rows={rows}"
            )
        _positive_importance(
            self.importance,
            expected=(experts, hidden),
            where=self.name,
        )
        if not bool(torch.isfinite(self.weight).all().item()):
            raise ValueError(f"{self.name}: packed gate/up contains non-finite values")
        return experts, gate_stop - gate_start, hidden


@dataclass(frozen=True)
class PackedDown:
    """One routed packed down tensor and post-SwiGLU input moments."""

    name: str
    weight: torch.Tensor
    importance: torch.Tensor

    def validate(
        self,
        *,
        experts: int,
        intermediate: int,
        hidden: int,
        group_size: int,
    ) -> None:
        if not self.name:
            raise ValueError("packed down name cannot be empty")
        expected_weight = (experts, hidden, intermediate)
        if tuple(self.weight.shape) != expected_weight:
            raise ValueError(
                f"{self.name}: packed down shape {tuple(self.weight.shape)} != "
                f"{expected_weight}"
            )
        if intermediate % group_size:
            raise ValueError(
                f"{self.name}: intermediate width {intermediate} is not "
                f"divisible by group_size={group_size}"
            )
        _positive_importance(
            self.importance,
            expected=(experts, intermediate),
            where=self.name,
        )
        if not bool(torch.isfinite(self.weight).all().item()):
            raise ValueError(f"{self.name}: packed down contains non-finite values")


def packed_post_norm_consumers(
    packed: PackedGateUp,
    *,
    config: PrismaSnapSearchConfig,
) -> list[PrismaSnapConsumer]:
    """Expand ``[E,2I,H]`` without collapsing the serving expert axis.

    Every expert slice gets a distinct consumer name and therefore a distinct
    static-6 global.  Flattening ``E`` into the row axis would instead choose
    one model-wide expert global and score a container the exporter never
    emits.
    """
    experts, _intermediate, _hidden = packed.validate(
        group_size=config.group_size
    )
    return [
        PrismaSnapConsumer(
            name=f"{packed.name}::expert:{expert}",
            weight=packed.weight[expert],
            importance=packed.importance[expert],
            mode="column_inverse",
        )
        for expert in range(experts)
    ]


def search_packed_post_norm_scale(
    packed: PackedGateUp,
    *,
    config: PrismaSnapSearchConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Search one post-norm scale over every routed expert objective."""
    consumers = packed_post_norm_consumers(packed, config=config)
    reference = packed.importance.to(torch.float64).mean(dim=0)
    scale, stats = search_diagonal_scale(consumers, reference, config=config)
    result = dict(stats)
    result.update(
        {
            "algorithm": PRISMASNAP_MOE_ALGORITHM,
            "objective_experts": int(packed.weight.shape[0]),
            "expert_global_scope": "per_expert_logical_tensor",
        }
    )
    return scale, result


def packed_up_down_consumers(
    gate_up: PackedGateUp,
    down: PackedDown,
    expert: int,
    *,
    config: PrismaSnapSearchConfig,
) -> list[PrismaSnapConsumer]:
    """Build a production-faithful objective for one packed expert seam.

    The gate rows are stationary but share their serving global with the up
    rows.  Giving both slices the same consumer name makes the dense objective
    take the maximum natural global across the two slices, exactly matching
    the exporter's per-expert joint gate/up global.  The down tensor has its
    own per-expert global.
    """
    experts, intermediate, hidden = gate_up.validate(group_size=config.group_size)
    down.validate(
        experts=experts,
        intermediate=intermediate,
        hidden=hidden,
        group_size=config.group_size,
    )
    if type(expert) is not int or expert < 0 or expert >= experts:
        raise ValueError(f"expert index {expert!r} is outside [0, {experts})")
    gate_start, gate_stop = gate_up.gate_rows
    up_start, up_stop = gate_up.up_rows
    logical_gate_up = f"{gate_up.name}::expert:{expert}"
    return [
        PrismaSnapConsumer(
            name=logical_gate_up,
            weight=gate_up.weight[expert, gate_start:gate_stop],
            importance=gate_up.importance[expert],
            mode="stationary",
        ),
        PrismaSnapConsumer(
            name=logical_gate_up,
            weight=gate_up.weight[expert, up_start:up_stop],
            importance=gate_up.importance[expert],
            mode="row",
        ),
        PrismaSnapConsumer(
            name=f"{down.name}::expert:{expert}",
            weight=down.weight[expert],
            importance=down.importance[expert],
            mode="column_inverse",
        ),
    ]


def search_packed_up_down_scales(
    gate_up: PackedGateUp,
    down: PackedDown,
    *,
    config: PrismaSnapSearchConfig,
    post_norm_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Search independent ``[E, I]`` up/down scales.

    Searches are intentionally expert-local because both route-conditioned
    importance and serving globals are expert-local.  The work remains CUDA
    tensor work; a future fused search may batch experts only if it reproduces
    these receipts bit-for-bit.
    """
    experts, intermediate, hidden = gate_up.validate(group_size=config.group_size)
    down.validate(
        experts=experts,
        intermediate=intermediate,
        hidden=hidden,
        group_size=config.group_size,
    )
    folded_scale: torch.Tensor | None = None
    if post_norm_scale is not None:
        if (
            post_norm_scale.ndim != 1
            or int(post_norm_scale.numel()) != hidden
            or not bool(torch.isfinite(post_norm_scale).all().item())
            or bool((post_norm_scale <= 0).any().item())
        ):
            raise ValueError("packed post-norm scale must be finite positive [H]")
        folded_scale = post_norm_scale

    scales: list[torch.Tensor] = []
    records: list[dict[str, object]] = []
    for expert in range(experts):
        expert_weight = gate_up.weight[expert]
        expert_importance = gate_up.importance[expert]
        if folded_scale is not None:
            # Materialization rounds the post-norm inverse fold back to source
            # BF16 before the expert-row fold. Search from those exact bytes
            # and move the activation moment into x' = d*x coordinates. Fold
            # one expert at a time: promoting a release-sized packed bank to
            # fp64 would create a tens-of-GB transient and violate residency.
            device_scale = folded_scale.to(
                device=expert_weight.device, dtype=torch.float64
            )
            expert_weight = (
                expert_weight.to(torch.float64) / device_scale.view(1, -1)
            ).to(expert_weight.dtype)
            expert_importance = (
                expert_importance.to(torch.float64) * device_scale.square()
            ).to(expert_importance.dtype)
        local_gate_up = PackedGateUp(
            name=gate_up.name,
            weight=expert_weight.unsqueeze(0),
            importance=expert_importance.unsqueeze(0),
            gate_rows=gate_up.gate_rows,
            up_rows=gate_up.up_rows,
        )
        local_down = PackedDown(
            name=down.name,
            weight=down.weight[expert].unsqueeze(0),
            importance=down.importance[expert].unsqueeze(0),
        )
        consumers = packed_up_down_consumers(
            local_gate_up, local_down, 0, config=config
        )
        scale, stats = search_diagonal_scale(
            consumers,
            down.importance[expert],
            config=config,
        )
        row = dict(stats)
        row.update(
            {
                "algorithm": PRISMASNAP_MOE_ALGORITHM,
                "expert": expert,
                "expert_global_scope": "per_expert_logical_tensor",
            }
        )
        scales.append(scale)
        records.append(row)
    return torch.stack(scales, dim=0), records


def fp64_router_and_expert_invariance(
    *,
    hidden: torch.Tensor,
    norm_gamma: torch.Tensor,
    norm_scale: torch.Tensor,
    router_weight: torch.Tensor,
    gate_up: torch.Tensor | Sequence[torch.Tensor | Sequence[torch.Tensor]],
    down: torch.Tensor | Sequence[torch.Tensor],
    expert_scales: torch.Tensor,
    top_k: int,
    shared_gate_up: Sequence[torch.Tensor] = (),
    shared_down: Sequence[torch.Tensor] = (),
    shared_output_gate_weight: Sequence[torch.Tensor] = (),
    shared_expert_scales: Sequence[torch.Tensor] = (),
) -> dict[str, float]:
    """Prove router logits, routing, and the complete MoE output in fp64.

    This is a deterministic algebra gate, not a substitute for served BF16
    fold KL.  ``gate_up`` is ``[E,2I,H]`` and ``down`` is ``[E,H,I]``.
    The reference uses top-k softmax routing and SwiGLU expert outputs. Each
    optional shared expert is always evaluated and multiplied by its typed
    scalar sigmoid output gate, matching the Qwen MoE residual branch.
    """
    tensors: Mapping[str, torch.Tensor] = {
        "hidden": hidden,
        "norm_gamma": norm_gamma,
        "norm_scale": norm_scale,
        "router_weight": router_weight,
        "expert_scales": expert_scales,
    }
    for name, value in tensors.items():
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be one finite tensor")
    if hidden.ndim != 2 or norm_gamma.ndim != 1 or norm_scale.ndim != 1:
        raise ValueError("hidden must be [T,H] and norm gamma/scale must be [H]")
    tokens, hidden_size = map(int, hidden.shape)
    if tuple(norm_gamma.shape) != (hidden_size,) or tuple(norm_scale.shape) != (
        hidden_size,
    ):
        raise ValueError("norm gamma/scale do not match hidden width")
    if router_weight.ndim != 2 or int(router_weight.shape[1]) != hidden_size:
        raise ValueError("router weight must be [E,H]")
    experts = int(router_weight.shape[0])
    routed_gate_bank: list[torch.Tensor] = []
    routed_up_bank: list[torch.Tensor] = []
    if torch.is_tensor(gate_up):
        if (
            gate_up.ndim != 3
            or int(gate_up.shape[0]) != experts
            or int(gate_up.shape[2]) != hidden_size
            or int(gate_up.shape[1]) % 2
            or not bool(torch.isfinite(gate_up).all().item())
        ):
            raise ValueError("gate_up must be finite [E,2I,H]")
        intermediate = int(gate_up.shape[1]) // 2
        for expert in range(experts):
            routed_gate_bank.append(gate_up[expert, :intermediate])
            routed_up_bank.append(gate_up[expert, intermediate:])
    else:
        if len(gate_up) != experts:
            raise ValueError("gate/up sequence must contain E experts")
        intermediate = -1
        for expert, value in enumerate(gate_up):
            if torch.is_tensor(value):
                if value.ndim != 2 or int(value.shape[0]) % 2:
                    raise ValueError(f"expert {expert} gate_up must be [2I,H]")
                split = int(value.shape[0]) // 2
                gate_value, up_value = value[:split], value[split:]
            elif isinstance(value, Sequence) and len(value) == 2:
                gate_value, up_value = value
                if not torch.is_tensor(gate_value) or not torch.is_tensor(up_value):
                    raise ValueError(f"expert {expert} gate/up pair must be tensors")
                split = int(gate_value.shape[0]) if gate_value.ndim == 2 else -1
            else:
                raise ValueError(f"expert {expert} gate_up entry is malformed")
            if (
                split <= 0
                or tuple(gate_value.shape) != (split, hidden_size)
                or tuple(up_value.shape) != (split, hidden_size)
                or not bool(torch.isfinite(gate_value).all().item())
                or not bool(torch.isfinite(up_value).all().item())
            ):
                raise ValueError(f"expert {expert} gate/up shapes differ")
            if intermediate < 0:
                intermediate = split
            elif intermediate != split:
                raise ValueError("routed expert intermediate widths differ")
            routed_gate_bank.append(gate_value)
            routed_up_bank.append(up_value)
    routed_down_bank: list[torch.Tensor] = []
    if torch.is_tensor(down):
        if (
            tuple(down.shape) != (experts, hidden_size, intermediate)
            or not bool(torch.isfinite(down).all().item())
        ):
            raise ValueError("down must be finite [E,H,I]")
        routed_down_bank.extend(down[expert] for expert in range(experts))
    else:
        if len(down) != experts:
            raise ValueError("down sequence must contain E experts")
        for expert, value in enumerate(down):
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != (hidden_size, intermediate)
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"expert {expert} down must be finite [H,I]")
            routed_down_bank.append(value)
    if tuple(expert_scales.shape) != (experts, intermediate):
        raise ValueError("expert scales must be [E,I]")
    if type(top_k) is not int or not 1 <= top_k <= experts:
        raise ValueError("top_k must be an integer in [1,E]")
    if tokens <= 0 or bool((norm_scale <= 0).any().item()) or bool(
        (expert_scales <= 0).any().item()
    ):
        raise ValueError("scale vectors must be positive and token count nonzero")
    shared_count = len(shared_gate_up)
    if not (
        len(shared_down)
        == len(shared_output_gate_weight)
        == len(shared_expert_scales)
        == shared_count
    ):
        raise ValueError("shared expert tensors/scales must have equal lengths")
    for index, (shared_gu, shared_dw, shared_output_gate, shared_scale) in enumerate(
        zip(
            shared_gate_up,
            shared_down,
            shared_output_gate_weight,
            shared_expert_scales,
        )
    ):
        shared_tensors = {
            "gate_up": shared_gu,
            "down": shared_dw,
            "output_gate": shared_output_gate,
            "scale": shared_scale,
        }
        if any(
            not torch.is_tensor(value)
            or not bool(torch.isfinite(value).all().item())
            for value in shared_tensors.values()
        ):
            raise ValueError(f"shared expert {index} tensors must be finite")
        if (
            shared_gu.ndim != 2
            or int(shared_gu.shape[1]) != hidden_size
            or int(shared_gu.shape[0]) % 2
        ):
            raise ValueError(f"shared expert {index} gate_up must be [2I,H]")
        shared_intermediate = int(shared_gu.shape[0]) // 2
        if tuple(shared_dw.shape) != (hidden_size, shared_intermediate):
            raise ValueError(f"shared expert {index} down must be [H,I]")
        if tuple(shared_output_gate.shape) != (1, hidden_size):
            raise ValueError(
                f"shared expert {index} output gate must be scalar [1,H]"
            )
        if tuple(shared_scale.shape) != (shared_intermediate,) or bool(
            (shared_scale <= 0).any().item()
        ):
            raise ValueError(f"shared expert {index} scale must be positive [I]")

    x = hidden.to(torch.float64)
    gamma = norm_gamma.to(device=x.device, dtype=torch.float64)
    d = norm_scale.to(device=x.device, dtype=torch.float64)
    router = router_weight.to(device=x.device, dtype=torch.float64)
    # Do not promote the full expert bank to fp64.  A release-sized layer can
    # hold hundreds of experts; the algebra gate only needs the deterministically
    # selected slices and must not create a second resident copy of the bank.
    routed_gate_bank = [value.to(device=x.device) for value in routed_gate_bank]
    routed_up_bank = [value.to(device=x.device) for value in routed_up_bank]
    routed_down_bank = [value.to(device=x.device) for value in routed_down_bank]
    u_scale = expert_scales.to(device=x.device, dtype=torch.float64)
    shared_gu_bank = [value.to(device=x.device) for value in shared_gate_up]
    shared_dw_bank = [value.to(device=x.device) for value in shared_down]
    shared_gate_bank = [
        value.to(device=x.device, dtype=torch.float64)
        for value in shared_output_gate_weight
    ]
    shared_scale_bank = [
        value.to(device=x.device, dtype=torch.float64)
        for value in shared_expert_scales
    ]

    normalized = x * gamma
    folded_normalized = x * (gamma * d)
    logits = normalized @ router.transpose(0, 1)
    folded_logits = folded_normalized @ (router / d.view(1, -1)).transpose(0, 1)
    values, indices = torch.topk(logits, k=top_k, dim=-1, sorted=True)
    folded_values, folded_indices = torch.topk(
        folded_logits, k=top_k, dim=-1, sorted=True
    )
    route_weights = torch.softmax(values, dim=-1)
    folded_route_weights = torch.softmax(folded_values, dim=-1)

    def routed_output(folded: bool) -> torch.Tensor:
        output = torch.zeros(tokens, hidden_size, device=x.device, dtype=torch.float64)
        routed_indices = folded_indices if folded else indices
        routed_weights = folded_route_weights if folded else route_weights
        for token in range(tokens):
            for slot in range(top_k):
                expert = int(routed_indices[token, slot].item())
                expert_input = folded_normalized[token] if folded else normalized[token]
                gate_rows = routed_gate_bank[expert].to(torch.float64)
                up_rows = routed_up_bank[expert].to(torch.float64)
                if folded:
                    gate_rows = gate_rows / d.view(1, -1)
                    up_rows = up_rows / d.view(1, -1)
                if folded:
                    up_rows = up_rows * u_scale[expert].view(-1, 1)
                gate = torch.nn.functional.silu(gate_rows @ expert_input)
                up = up_rows @ expert_input
                down_base = routed_down_bank[expert].to(torch.float64)
                down_matrix = (
                    down_base / u_scale[expert].view(1, -1)
                    if folded
                    else down_base
                )
                output[token] += routed_weights[token, slot] * (
                    down_matrix @ (gate * up)
                )
        expert_input = folded_normalized if folded else normalized
        for shared_gu, shared_dw, output_gate, shared_scale in zip(
            shared_gu_bank,
            shared_dw_bank,
            shared_gate_bank,
            shared_scale_bank,
        ):
            matrix = shared_gu.to(torch.float64)
            matrix = matrix / d.view(1, -1) if folded else matrix
            shared_intermediate = int(matrix.shape[0]) // 2
            gate_rows = matrix[:shared_intermediate]
            up_rows = matrix[shared_intermediate:]
            if folded:
                up_rows = up_rows * shared_scale.view(-1, 1)
            down_matrix = shared_dw.to(torch.float64)
            if folded:
                down_matrix = down_matrix / shared_scale.view(1, -1)
            scalar_gate_matrix = output_gate / d.view(1, -1) if folded else output_gate
            scalar_gate = torch.sigmoid(
                expert_input @ scalar_gate_matrix.transpose(0, 1)
            )
            gate = torch.nn.functional.silu(
                expert_input @ gate_rows.transpose(0, 1)
            )
            up = expert_input @ up_rows.transpose(0, 1)
            output += scalar_gate * ((gate * up) @ down_matrix.transpose(0, 1))
        return output

    before_output = routed_output(False)
    after_output = routed_output(True)
    logit_error = float((logits - folded_logits).abs().max().item())
    route_weight_error = float(
        (route_weights - folded_route_weights).abs().max().item()
    )
    output_error = float((before_output - after_output).abs().max().item())
    routing_changed = float((indices != folded_indices).sum().item())
    result = {
        "router_logit_max_abs": logit_error,
        "route_weight_max_abs": route_weight_error,
        "routed_output_max_abs": output_error,
        "routing_changed": routing_changed,
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise RuntimeError("PrismaSnap MoE fp64 invariance became non-finite")
    if routing_changed != 0.0 or max(
        logit_error, route_weight_error, output_error
    ) > 1e-10:
        raise RuntimeError(f"PrismaSnap MoE fp64 invariance failed: {result}")
    return result


def apply_packed_expert_slice_transform(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    *,
    expert_axis: int,
    channel_axis: int,
    expert: int | None,
    channel_start: int | None,
    channel_stop: int | None,
    operation: str,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply one schema-declared packed/per-expert materialization slice.

    ``expert=None`` broadcasts a ``[E,C]`` scale over every expert.  An
    integer expert applies a 1-D scale to exactly that expert.  Optional
    channel bounds select the up half of a fused gate/up row axis.  No axis or
    slice is guessed from the tensor name.
    """
    if tensor.ndim not in {2, 3}:
        raise ValueError("MoE slice transform supports rank-2/rank-3 tensors")
    ndim = tensor.ndim
    expert_axis %= ndim
    channel_axis %= ndim
    if expert_axis == channel_axis:
        raise ValueError("expert and channel axes must differ")
    if expert is None:
        if tensor.ndim != 3 or scale.ndim != 2:
            raise ValueError("broadcast expert transform requires rank-3 tensor/[E,C] scale")
        if int(scale.shape[0]) != int(tensor.shape[expert_axis]):
            raise ValueError("expert scale count does not match tensor expert axis")
        expert_indices = range(int(tensor.shape[expert_axis]))
    else:
        if type(expert) is not int or not 0 <= expert < int(tensor.shape[expert_axis]):
            raise ValueError("expert slice index is out of range")
        if scale.ndim != 1:
            raise ValueError("one expert slice requires a rank-1 scale")
        expert_indices = (expert,)
    size = int(tensor.shape[channel_axis])
    start = 0 if channel_start is None else channel_start
    stop = size if channel_stop is None else channel_stop
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= size:
        raise ValueError("channel slice bounds are malformed")
    expected_channels = stop - start
    if int(scale.shape[-1]) != expected_channels:
        raise ValueError("scale width does not match selected channel slice")
    if operation not in {"multiply", "divide"}:
        raise ValueError(f"unsupported MoE slice operation {operation!r}")

    value = tensor.to(torch.float64).clone()
    for expert_index in expert_indices:
        selector = [slice(None)] * ndim
        selector[expert_axis] = expert_index
        selector[channel_axis] = slice(start, stop)
        factor = (
            scale[expert_index] if expert is None else scale
        ).to(device=value.device, dtype=torch.float64)
        view_shape = [1] * ndim
        # Indexing the expert axis removes one dimension.  Map the original
        # channel axis into the selected view's coordinates.
        view_channel_axis = channel_axis - (1 if expert_axis < channel_axis else 0)
        selected = value[tuple(selector)]
        factor_shape = [1] * selected.ndim
        factor_shape[view_channel_axis] = expected_channels
        factor = factor.reshape(factor_shape)
        value[tuple(selector)] = (
            selected * factor if operation == "multiply" else selected / factor
        )
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError("MoE slice transform produced non-finite values")
    result = value.to(output_dtype or tensor.dtype).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("MoE slice transform overflowed output dtype")
    return result


__all__ = [
    "PRISMASNAP_MOE_ALGORITHM",
    "PRISMASNAP_MOE_PROMOTION",
    "PackedDown",
    "PackedGateUp",
    "apply_packed_expert_slice_transform",
    "fp64_router_and_expert_invariance",
    "moe_search_contract",
    "packed_post_norm_consumers",
    "packed_up_down_consumers",
    "search_packed_post_norm_scale",
    "search_packed_up_down_scales",
]
