"""Exact 256-state tail-biting encoder for PrismaQuant trellis wires.

This is the production-owned promotion of the Stage-5 research encoder.  All
arguments that can affect encoded bits are explicit at the render-plan layer;
there are no environment-derived encoder defaults here.  The eager route is
the executable CPU reference.  CUDA uses four fused Triton launches per
chunk (warm metrics, candidate closure, survivors, traceback), keeping cache
fill GPU-bound.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .trellis_formats import (
    E2M1_FAMILY,
    E4M3_FAMILY,
    E4M3FN_NAN_CODES,
    GENERATOR_OCTAL,
    MIN_TRELLIS_STEPS,
    STATE_COUNT,
    SUPERBLOCK_WEIGHTS,
    get_trellis_family,
    native_code_value,
    validate_alphabets,
)

try:  # CPU reference installs do not need Triton.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the local torch image
    triton = None
    tl = None


BIG = 1.0e9
TAILBITE_CANDIDATES = 4
E4M3_MAX = 448.0
POINT_WINDOW = 16
_E2M1_BYPASS_CODES = (15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7)
_E2M1_BYPASS_VALUES = tuple(
    native_code_value(E2M1_FAMILY, code) for code in _E2M1_BYPASS_CODES
)


class TrellisEncoderError(RuntimeError):
    """The requested encode cannot produce the priced wire exactly."""


@dataclass(frozen=True, slots=True)
class EncodedTrellisPlanes:
    reconstruction: torch.Tensor
    u_bits: torch.Tensor
    point_indices: torch.Tensor
    bypass_codes: torch.Tensor
    scale_blob: bytes
    global_scale_real: float


@dataclass(frozen=True, slots=True)
class _PointTables:
    trellis_positions: torch.Tensor
    bypass_positions: torch.Tensor
    points: torch.Tensor
    point_counts: torch.Tensor
    rates: torch.Tensor


def encoder_source_sha256() -> str:
    """Hash the checked-in encoder source, never bytecode/cache files."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_trellis() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build predecessor, input-bit, and convolutional subset tables."""
    g0, g1 = (int(value, 8) for value in GENERATOR_OCTAL)
    memory = int(math.log2(STATE_COUNT))
    mask = STATE_COUNT - 1
    previous = torch.empty(STATE_COUNT, 2, dtype=torch.long)
    input_bit = torch.empty(STATE_COUNT, 2, dtype=torch.long)
    subset = torch.empty(STATE_COUNT, 2, dtype=torch.long)
    for next_state in range(STATE_COUNT):
        for branch in (0, 1):
            register = (next_state << 1) | branch
            previous[next_state, branch] = register & mask
            input_bit[next_state, branch] = register >> memory
            subset[next_state, branch] = (
                2 * ((register & g0).bit_count() & 1)
                + ((register & g1).bit_count() & 1)
            )
    return previous, input_bit, subset


def _point_tables(
    schedule: Sequence[int],
    alphabets: Mapping[int, Sequence[int]],
    *,
    family: str,
    device: torch.device,
) -> _PointTables:
    spec = get_trellis_family(family)
    rates = tuple(int(value) for value in schedule)
    if len(rates) != SUPERBLOCK_WEIGHTS:
        raise TrellisEncoderError("encoder block schedule must have 256 entries")
    schedule_tensor = torch.tensor(rates, dtype=torch.long)
    trellis_positions = torch.nonzero(
        schedule_tensor < spec.bypass_rate
    ).flatten()
    bypass_positions = torch.nonzero(
        schedule_tensor == spec.bypass_rate
    ).flatten()
    if int(trellis_positions.numel()) < MIN_TRELLIS_STEPS:
        raise TrellisEncoderError(
            f"encoder block has {int(trellis_positions.numel())} shaped "
            f"positions; at least {MIN_TRELLIS_STEPS} are required"
        )
    maximum_points = max(
        1 << (rates[int(position)] - 1)
        for position in trellis_positions.tolist()
    )
    points = torch.zeros(
        int(trellis_positions.numel()),
        4,
        maximum_points,
        dtype=torch.float32,
    )
    counts = torch.empty(int(trellis_positions.numel()), dtype=torch.long)
    shaped_rates = torch.empty_like(counts)
    for table_index, position in enumerate(trellis_positions.tolist()):
        rate = rates[position]
        codes = tuple(int(code) for code in alphabets[rate])
        count = 1 << (rate - 1)
        counts[table_index] = count
        shaped_rates[table_index] = rate
        for subset in range(4):
            subset_codes = codes[subset::4]
            if len(subset_codes) != count:
                raise TrellisEncoderError(
                    f"rate-{rate} alphabet partition is unbalanced"
                )
            values = torch.tensor(
                [native_code_value(spec, code) for code in subset_codes],
                dtype=torch.float32,
            )
            points[table_index, subset, :count] = values
            points[table_index, subset, count:] = values[0]
    return _PointTables(
        trellis_positions.to(device),
        bypass_positions.to(device),
        points.to(device),
        counts.to(device),
        shaped_rates.to(device),
    )


def _point_costs_full(
    x: torch.Tensor,
    objective_weight: torch.Tensor,
    tables: _PointTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance = (
        (x.unsqueeze(-1).unsqueeze(-1) - tables.points.unsqueeze(0)).pow(2)
        * objective_weight.unsqueeze(-1).unsqueeze(-1)
    )
    cost, point = distance.min(dim=-1)
    point = torch.where(
        point >= tables.point_counts.view(1, -1, 1),
        torch.zeros_like(point),
        point,
    )
    cost = cost / cost.mean().clamp_min(1.0e-30)
    return cost.contiguous(), point.to(torch.uint8).contiguous()


def _point_costs_windowed(
    x: torch.Tensor,
    objective_weight: torch.Tensor,
    tables: _PointTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact nearest point with a bounded insertion-point window."""
    batch, steps = x.shape
    cost = torch.empty(batch, steps, 4, dtype=torch.float32, device=x.device)
    point = torch.empty(batch, steps, 4, dtype=torch.uint8, device=x.device)
    for rate in sorted(set(int(value) for value in tables.rates.tolist())):
        positions = torch.nonzero(tables.rates == rate).flatten()
        target = x.index_select(1, positions)
        weight = objective_weight.index_select(1, positions)
        count = 1 << (rate - 1)
        for subset in range(4):
            levels = tables.points[positions[0], subset, :count]
            if count <= POINT_WINDOW:
                distance = (target.unsqueeze(-1) - levels).pow(2)
                minimum, index = distance.min(dim=-1)
            else:
                insertion = torch.bucketize(target.contiguous(), levels)
                first = insertion - POINT_WINDOW // 2
                offsets = torch.arange(POINT_WINDOW, device=x.device)
                candidates = (first.unsqueeze(-1) + offsets).clamp(
                    0, count - 1
                )
                values = levels[candidates]
                distance = (target.unsqueeze(-1) - values).pow(2)
                minimum, local = distance.min(dim=-1)
                index = candidates.gather(-1, local.unsqueeze(-1)).squeeze(-1)
            cost[:, positions, subset] = minimum * weight
            point[:, positions, subset] = index.to(torch.uint8)
    cost = cost / cost.mean().clamp_min(1.0e-30)
    return cost.contiguous(), point.contiguous()


def _viterbi_forward_eager(
    cost: torch.Tensor,
    previous: torch.Tensor,
    subset: torch.Tensor,
    initial: torch.Tensor,
    *,
    keep_survivors: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch, steps, _ = cost.shape
    metrics = initial
    offset = torch.zeros(
        *initial.shape[:-1], 1, dtype=cost.dtype, device=cost.device
    )
    previous_flat = previous.reshape(-1)
    subset_flat = subset.reshape(-1)
    survivors = (
        torch.empty(
            *initial.shape[:-1],
            steps,
            STATE_COUNT,
            dtype=torch.uint8,
            device=cost.device,
        )
        if keep_survivors
        else None
    )
    for time_index in range(steps):
        branch_cost = cost[:, time_index].index_select(1, subset_flat).view(
            batch, STATE_COUNT, 2
        )
        if initial.ndim == 2:
            candidate = metrics.index_select(1, previous_flat).view(
                batch, STATE_COUNT, 2
            ) + branch_cost
        else:
            candidates = int(initial.shape[1])
            candidate = metrics.index_select(2, previous_flat).view(
                batch, candidates, STATE_COUNT, 2
            ) + branch_cost.unsqueeze(1)
        metrics, branch = candidate.min(dim=-1)
        if survivors is not None:
            survivors[..., time_index, :] = branch.to(torch.uint8)
        minimum = metrics.min(dim=-1, keepdim=True).values
        metrics = metrics - minimum
        offset = offset + minimum
    return metrics, offset, survivors


def _tailbite_eager(
    cost: torch.Tensor,
    point_index: torch.Tensor,
    points: torch.Tensor,
    previous: torch.Tensor,
    input_bit: torch.Tensor,
    subset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, steps, _ = cost.shape
    metrics = torch.zeros(batch, STATE_COUNT, dtype=cost.dtype, device=cost.device)
    for _ in range(2):
        metrics, _, _ = _viterbi_forward_eager(
            cost, previous, subset, metrics, keep_survivors=False
        )
    candidates = metrics.topk(
        TAILBITE_CANDIDATES, dim=1, largest=False
    ).indices
    initial = torch.full(
        (batch, TAILBITE_CANDIDATES, STATE_COUNT),
        BIG,
        dtype=cost.dtype,
        device=cost.device,
    )
    initial.scatter_(2, candidates.unsqueeze(-1), 0.0)
    closing_metric, closing_offset, _ = _viterbi_forward_eager(
        cost, previous, subset, initial, keep_survivors=False
    )
    closing = (closing_metric + closing_offset).gather(
        2, candidates.unsqueeze(-1)
    ).squeeze(-1)
    start = candidates.gather(1, closing.argmin(dim=1, keepdim=True))
    initial = torch.full(
        (batch, STATE_COUNT), BIG, dtype=cost.dtype, device=cost.device
    )
    initial.scatter_(1, start, 0.0)
    _, _, survivors = _viterbi_forward_eager(
        cost, previous, subset, initial, keep_survivors=True
    )
    assert survivors is not None
    state = start.squeeze(1)
    reconstruction = torch.empty(
        batch, steps, dtype=torch.float32, device=cost.device
    )
    coded = torch.empty(batch, steps, dtype=torch.uint8, device=cost.device)
    point = torch.empty_like(coded)
    rows = torch.arange(batch, device=cost.device)
    maximum_points = points.shape[-1]
    flat_points = points.reshape(-1)
    for time_index in range(steps - 1, -1, -1):
        branch = survivors[:, time_index].gather(
            1, state[:, None]
        ).squeeze(1).long()
        label = subset[state, branch]
        selected_point = point_index[rows, time_index, label].long()
        reconstruction[:, time_index] = flat_points[
            (time_index * 4 + label) * maximum_points + selected_point
        ]
        coded[:, time_index] = input_bit[state, branch].to(torch.uint8)
        point[:, time_index] = selected_point.to(torch.uint8)
        state = previous[state, branch]
    if not torch.equal(state, start.squeeze(1)):
        raise AssertionError("tail-biting constraint violated")
    return reconstruction, coded, point


if triton is not None:

    @triton.jit
    def _warm_metrics_kernel(
        cost_ptr, previous_ptr, subset_ptr, metrics_ptr,
        steps: tl.constexpr, states: tl.constexpr,
    ):
        block = tl.program_id(0)
        state = tl.arange(0, states)
        table = state * 2
        previous0 = tl.load(previous_ptr + table)
        previous1 = tl.load(previous_ptr + table + 1)
        subset0 = tl.load(subset_ptr + table)
        subset1 = tl.load(subset_ptr + table + 1)
        metric = tl.zeros((states,), tl.float32)
        for scan_time in tl.range(0, 2 * steps, loop_unroll_factor=1):
            time_index = scan_time % steps
            base = (block * steps + time_index) * 4
            left = tl.gather(metric, previous0, 0) + tl.load(
                cost_ptr + base + subset0
            )
            right = tl.gather(metric, previous1, 0) + tl.load(
                cost_ptr + base + subset1
            )
            metric = tl.where(right < left, right, left)
            metric -= tl.min(metric, axis=0)
        tl.store(metrics_ptr + block * states + state, metric)


    @triton.jit
    def _candidate_kernel(
        cost_ptr, candidates_ptr, previous_ptr, subset_ptr, closing_ptr,
        candidates: tl.constexpr, steps: tl.constexpr, states: tl.constexpr,
    ):
        path = tl.program_id(0)
        block = path // candidates
        candidate_index = path - block * candidates
        state = tl.arange(0, states)
        start = tl.load(candidates_ptr + block * candidates + candidate_index)
        table = state * 2
        previous0 = tl.load(previous_ptr + table)
        previous1 = tl.load(previous_ptr + table + 1)
        subset0 = tl.load(subset_ptr + table)
        subset1 = tl.load(subset_ptr + table + 1)
        metric = tl.where(state == start, 0.0, 1.0e9)
        offset = 0.0
        for time_index in tl.range(0, steps, loop_unroll_factor=1):
            base = (block * steps + time_index) * 4
            left = tl.gather(metric, previous0, 0) + tl.load(
                cost_ptr + base + subset0
            )
            right = tl.gather(metric, previous1, 0) + tl.load(
                cost_ptr + base + subset1
            )
            metric = tl.where(right < left, right, left)
            minimum = tl.min(metric, axis=0)
            metric -= minimum
            offset += minimum
        at_start = tl.sum(tl.where(state == start, metric, 0.0), axis=0)
        tl.store(
            closing_ptr + block * candidates + candidate_index,
            at_start + offset,
        )


    @triton.jit
    def _survivor_kernel(
        cost_ptr, start_ptr, previous_ptr, subset_ptr, survivor_ptr,
        steps: tl.constexpr, states: tl.constexpr,
    ):
        block = tl.program_id(0)
        state = tl.arange(0, states)
        start = tl.load(start_ptr + block)
        table = state * 2
        previous0 = tl.load(previous_ptr + table)
        previous1 = tl.load(previous_ptr + table + 1)
        subset0 = tl.load(subset_ptr + table)
        subset1 = tl.load(subset_ptr + table + 1)
        metric = tl.where(state == start, 0.0, 1.0e9)
        for time_index in tl.range(0, steps, loop_unroll_factor=1):
            base = (block * steps + time_index) * 4
            left = tl.gather(metric, previous0, 0) + tl.load(
                cost_ptr + base + subset0
            )
            right = tl.gather(metric, previous1, 0) + tl.load(
                cost_ptr + base + subset1
            )
            choose_right = right < left
            metric = tl.where(choose_right, right, left)
            tl.store(
                survivor_ptr + (block * steps + time_index) * states + state,
                choose_right.to(tl.uint8),
            )
            metric -= tl.min(metric, axis=0)


    @triton.jit
    def _traceback_kernel(
        survivor_ptr, start_ptr, previous_ptr, input_ptr, subset_ptr,
        point_index_ptr, points_ptr, reconstruction_ptr, coded_ptr, point_ptr,
        tailbite_ptr, blocks, steps: tl.constexpr, states: tl.constexpr,
        maximum_points: tl.constexpr, block_batch: tl.constexpr,
    ):
        block = tl.program_id(0) * block_batch + tl.arange(0, block_batch)
        valid = block < blocks
        block64 = block.to(tl.int64)
        state = tl.load(start_ptr + block, mask=valid, other=0).to(tl.int64)
        initial = state
        for reverse_time in tl.range(0, steps, loop_unroll_factor=1):
            time_index = steps - 1 - reverse_time
            branch = tl.load(
                survivor_ptr + (block64 * steps + time_index) * states + state,
                mask=valid,
                other=0,
            ).to(tl.int64)
            table = state * 2 + branch
            label = tl.load(subset_ptr + table, mask=valid, other=0).to(tl.int64)
            point_index = tl.load(
                point_index_ptr + (block64 * steps + time_index) * 4 + label,
                mask=valid,
                other=0,
            ).to(tl.int64)
            value = tl.load(
                points_ptr
                + (time_index * 4 + label) * maximum_points
                + point_index,
                mask=valid,
                other=0.0,
            )
            tl.store(
                reconstruction_ptr + block64 * steps + time_index,
                value,
                mask=valid,
            )
            tl.store(
                coded_ptr + block64 * steps + time_index,
                tl.load(input_ptr + table, mask=valid, other=0).to(tl.uint8),
                mask=valid,
            )
            tl.store(
                point_ptr + block64 * steps + time_index,
                point_index.to(tl.uint8),
                mask=valid,
            )
            state = tl.load(previous_ptr + table, mask=valid, other=0).to(
                tl.int64
            )
        tl.store(tailbite_ptr + block, state == initial, mask=valid)


def _tailbite_triton(
    cost: torch.Tensor,
    point_index: torch.Tensor,
    points: torch.Tensor,
    previous: torch.Tensor,
    input_bit: torch.Tensor,
    subset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise TrellisEncoderError("Triton is not installed")
    if cost.device.type != "cuda":
        raise TrellisEncoderError("the Triton encoder requires CUDA")
    batch, steps, _ = cost.shape
    previous_i32 = previous.to(torch.int32).contiguous()
    input_u8 = input_bit.to(torch.uint8).contiguous()
    subset_i32 = subset.to(torch.int32).contiguous()
    metrics = torch.empty(
        batch, STATE_COUNT, dtype=torch.float32, device=cost.device
    )
    _warm_metrics_kernel[(batch,)](
        cost,
        previous_i32,
        subset_i32,
        metrics,
        steps=steps,
        states=STATE_COUNT,
        num_warps=8,
    )
    candidates = metrics.topk(
        TAILBITE_CANDIDATES, dim=1, largest=False
    ).indices.contiguous()
    closing = torch.empty(
        batch, TAILBITE_CANDIDATES, dtype=torch.float32, device=cost.device
    )
    _candidate_kernel[(batch * TAILBITE_CANDIDATES,)](
        cost,
        candidates,
        previous_i32,
        subset_i32,
        closing,
        candidates=TAILBITE_CANDIDATES,
        steps=steps,
        states=STATE_COUNT,
        num_warps=8,
    )
    start = candidates.gather(
        1, closing.argmin(dim=1, keepdim=True)
    ).squeeze(1).contiguous()
    survivors = torch.empty(
        batch, steps, STATE_COUNT, dtype=torch.uint8, device=cost.device
    )
    _survivor_kernel[(batch,)](
        cost,
        start,
        previous_i32,
        subset_i32,
        survivors,
        steps=steps,
        states=STATE_COUNT,
        num_warps=8,
    )
    reconstruction = torch.empty(
        batch, steps, dtype=torch.float32, device=cost.device
    )
    coded = torch.empty(batch, steps, dtype=torch.uint8, device=cost.device)
    point = torch.empty_like(coded)
    tailbite_ok = torch.empty(batch, dtype=torch.uint8, device=cost.device)
    block_batch = 128
    _traceback_kernel[(triton.cdiv(batch, block_batch),)](
        survivors,
        start,
        previous_i32,
        input_u8,
        subset_i32,
        point_index,
        points,
        reconstruction,
        coded,
        point,
        tailbite_ok,
        batch,
        steps=steps,
        states=STATE_COUNT,
        maximum_points=points.shape[-1],
        block_batch=block_batch,
        num_warps=4,
    )
    if not bool(tailbite_ok.bool().all().item()):
        raise AssertionError("tail-biting constraint violated")
    return reconstruction, coded, point


def _nearest_levels(
    x: torch.Tensor, levels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    insertion = torch.bucketize(x.contiguous(), levels)
    low_index = (insertion - 1).clamp(0, levels.numel() - 1)
    high_index = insertion.clamp(max=levels.numel() - 1)
    low = levels[low_index]
    high = levels[high_index]
    take_high = (high - x).abs() < (x - low).abs()
    return (
        torch.where(take_high, high, low),
        torch.where(take_high, high_index, low_index),
    )


def _scale_context(
    weight: torch.Tensor,
    col_weights: torch.Tensor,
    *,
    family: str,
    scale_rule: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bytes, float]:
    """Return normalized weight, objective, per-element scale, wire plane."""
    rows, columns = map(int, weight.shape)
    if scale_rule != "static_6" and family == E2M1_FAMILY:
        raise TrellisEncoderError(
            f"unsupported E2M1 trellis scale rule {scale_rule!r}; only "
            "the measured static_6 group-16 plane has a writer"
        )
    source = weight.detach().to(torch.float32)
    importance = col_weights.reshape(1, columns).to(
        device=weight.device, dtype=torch.float32
    ).clamp_min(1.0e-12)
    if family == E2M1_FAMILY:
        grouped = source.reshape(rows, columns // 16, 16)
        real_scale = grouped.abs().amax(dim=-1).clamp_min(1.0e-12) / 6.0
        global_scale = (real_scale.amax() / E4M3_MAX).clamp_min(1.0e-12)
        fp8_scale = (real_scale / global_scale).clamp(
            0.0, E4M3_MAX
        ).to(torch.float8_e4m3fn)
        effective = (fp8_scale.to(torch.float32) * global_scale).clamp_min(
            1.0e-12
        )
        scale_blob = (
            fp8_scale.view(torch.uint8)
            .detach()
            .to(device="cpu")
            .contiguous()
            .numpy()
            .tobytes()
        )
        global_real = float(global_scale.item())
        per_element = effective.repeat_interleave(16, dim=1)
    else:
        if scale_rule != "row_fp32_amax_448":
            raise TrellisEncoderError(
                f"unsupported E4M3 trellis scale rule {scale_rule!r}; only "
                "the measured per-row fp32 amax/448 plane has a writer"
            )
        live = source.abs().amax(dim=1, keepdim=True) / E4M3_MAX
        positive = live[live > 0]
        floor = (
            positive.min()
            if positive.numel()
            else torch.ones((), dtype=torch.float32, device=weight.device)
        )
        row_scale = torch.where(live > 0, live, floor.expand_as(live)).to(
            torch.float32
        )
        if float((source / row_scale).abs().amax().item()) > E4M3_MAX + 1e-3:
            raise TrellisEncoderError("E4M3 row normalization exceeds 448")
        per_element = row_scale.expand(rows, columns).contiguous()
        scale_blob = (
            row_scale.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
            .tobytes()
        )
        global_real = 1.0
    normalized = source / per_element
    objective = importance * per_element.pow(2)
    return normalized, objective, per_element, scale_blob, global_real


@contextmanager
def _determinism_mode(mode: str):
    if mode not in {"on", "off"}:
        raise TrellisEncoderError("determinism mode must be 'on' or 'off'")
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else False
    )
    torch.use_deterministic_algorithms(mode == "on", warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )


def encode_trellis_planes(
    weight: torch.Tensor,
    col_weights: torch.Tensor,
    *,
    family: str,
    schedule: Sequence[int],
    alphabets: Mapping[int, Sequence[int]],
    scale_rule: str,
    sb_chunk: int,
    determinism_mode: str,
    tailbite_candidates: int,
    backend: str,
    point_route: str,
) -> EncodedTrellisPlanes:
    """Encode one dense Linear and return the exact wire planes."""
    spec = get_trellis_family(family)
    if weight.ndim != 2:
        raise TrellisEncoderError("trellis encoder supports dense rank-2 weights")
    rows, columns = map(int, weight.shape)
    if columns % SUPERBLOCK_WEIGHTS:
        raise TrellisEncoderError(
            "trellis encoder requires input columns divisible by 256"
        )
    if len(schedule) != columns:
        raise TrellisEncoderError("trellis schedule width differs from weight")
    checked_alphabets = validate_alphabets(spec, schedule, alphabets)
    if col_weights is None or col_weights.numel() != columns:
        raise TrellisEncoderError(
            f"trellis col_weights must have shape ({columns},)"
        )
    vector = col_weights.detach().reshape(-1).to(torch.float32)
    if not bool(torch.isfinite(vector).all().item()) or bool(
        (vector < 0).any().item()
    ):
        raise TrellisEncoderError("trellis col_weights must be finite/nonnegative")
    if float(vector.sum().item()) <= 0.0:
        raise TrellisEncoderError("trellis col_weights must contain positive mass")
    if int(sb_chunk) < 1:
        raise TrellisEncoderError("encoder sb_chunk must be positive")
    if int(tailbite_candidates) != TAILBITE_CANDIDATES:
        raise TrellisEncoderError(
            f"only {TAILBITE_CANDIDATES} tail-biting candidates are "
            "bit-exactness qualified"
        )
    if backend not in {"eager", "triton"}:
        raise TrellisEncoderError("encoder backend must be eager or triton")
    if backend == "triton" and weight.device.type != "cuda":
        raise TrellisEncoderError("the Triton encoder requires CUDA")
    if point_route not in {"full", "windowed"}:
        raise TrellisEncoderError("point route must be full or windowed")

    normalized, objective, per_element, scale_blob, global_real = _scale_context(
        weight, vector, family=spec.family, scale_rule=scale_rule
    )
    reconstructed_normalized = torch.empty_like(normalized)
    u_bits = torch.zeros(rows, columns, dtype=torch.uint8, device=weight.device)
    point_indices = torch.zeros_like(u_bits)
    bypass_codes = torch.zeros_like(u_bits)
    previous, input_bit, subset = build_trellis()
    previous = previous.to(weight.device)
    input_bit = input_bit.to(weight.device)
    subset = subset.to(weight.device)

    with _determinism_mode(determinism_mode):
        for block, first_column in enumerate(
            range(0, columns, SUPERBLOCK_WEIGHTS)
        ):
            stop_column = first_column + SUPERBLOCK_WEIGHTS
            block_schedule = tuple(schedule[first_column:stop_column])
            tables = _point_tables(
                block_schedule,
                checked_alphabets,
                family=spec.family,
                device=weight.device,
            )
            block_x = normalized[:, first_column:stop_column]
            block_objective = objective[:, first_column:stop_column]
            shaped_x = block_x.index_select(1, tables.trellis_positions)
            shaped_objective = block_objective.index_select(
                1, tables.trellis_positions
            )
            for first_row in range(0, rows, int(sb_chunk)):
                stop_row = min(rows, first_row + int(sb_chunk))
                if point_route == "full":
                    cost, point = _point_costs_full(
                        shaped_x[first_row:stop_row],
                        shaped_objective[first_row:stop_row],
                        tables,
                    )
                else:
                    cost, point = _point_costs_windowed(
                        shaped_x[first_row:stop_row],
                        shaped_objective[first_row:stop_row],
                        tables,
                    )
                encoded = (
                    _tailbite_triton(
                        cost,
                        point,
                        tables.points,
                        previous,
                        input_bit,
                        subset,
                    )
                    if backend == "triton"
                    else _tailbite_eager(
                        cost,
                        point,
                        tables.points,
                        previous,
                        input_bit,
                        subset,
                    )
                )
                reconstruction, coded, selected_point = encoded
                target_columns = first_column + tables.trellis_positions
                reconstructed_normalized[
                    first_row:stop_row, target_columns
                ] = reconstruction
                u_bits[first_row:stop_row, target_columns] = coded
                point_indices[first_row:stop_row, target_columns] = selected_point

            if tables.bypass_positions.numel():
                bypass_x = block_x.index_select(1, tables.bypass_positions)
                if spec.family == E4M3_FAMILY:
                    bypass_value = bypass_x.clamp(
                        -E4M3_MAX, E4M3_MAX
                    ).to(torch.float8_e4m3fn).to(torch.float32)
                    codes = bypass_value.to(torch.float8_e4m3fn).view(torch.uint8)
                    if bool(
                        ((codes == 0x7F) | (codes == 0xFF)).any().item()
                    ):
                        raise TrellisEncoderError(
                            "E4M3 bypass produced a NaN native code"
                        )
                else:
                    levels = torch.tensor(
                        _E2M1_BYPASS_VALUES,
                        dtype=torch.float32,
                        device=weight.device,
                    )
                    bypass_value, ordinal = _nearest_levels(bypass_x, levels)
                    code_table = torch.tensor(
                        _E2M1_BYPASS_CODES,
                        dtype=torch.uint8,
                        device=weight.device,
                    )
                    codes = code_table[ordinal]
                target_columns = first_column + tables.bypass_positions
                reconstructed_normalized[:, target_columns] = bypass_value
                bypass_codes[:, target_columns] = codes

    reconstruction = (reconstructed_normalized * per_element).contiguous()
    return EncodedTrellisPlanes(
        reconstruction=reconstruction,
        u_bits=u_bits,
        point_indices=point_indices,
        bypass_codes=bypass_codes,
        scale_blob=scale_blob,
        global_scale_real=global_real,
    )


__all__ = [
    "EncodedTrellisPlanes",
    "TrellisEncoderError",
    "build_trellis",
    "encode_trellis_planes",
    "encoder_source_sha256",
]
