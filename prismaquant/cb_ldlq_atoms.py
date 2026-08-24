# SPDX-License-Identifier: Apache-2.0
"""Fixed-scale product-codebook LDLQ atoms for Gridbook CB fields.

Its outer-buffer recurrence comes from PrismaQuant's Apache-2.0 GPTQ
implementation; the dense
atom metric is derived directly from ``Z @ U_AA = residual``.  GLQ and the
QuIP family were inspection/oracle-only and supplied no source expression.

The serialized product atom is four columns for the NVFP4 family and two
columns for the FP8 family.  ``outer_tile_columns`` is only a buffering/GEMM
boundary and defaults to 64 for compatibility with the existing exporter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .cb_compile_contract import (
    ATOM_BATCHED_CHUNK_BEST,
    ATOM_CHUNK_BEST,
    CBCompileContractError,
    cb_compile_fail_closed,
    compile_cb_callable,
    refuse_cb_compile_fallback,
)
from .cb_layout import FP4_GROUP, VEC_DIM


LDLQ_OUTER_TILE_COLUMNS = 64
LDLQ_CANDIDATE_WORKSPACE_ENV = "PRISMAQUANT_CB_LDLQ_CANDIDATE_WORKSPACE_BYTES"
LDLQ_CANDIDATE_WORKSPACE_DEFAULT = 128 * 1024 * 1024
LDLQ_CANDIDATE_CHUNK_CAP = 4096


class CBLDLQError(RuntimeError):
    """Base exception for product-codebook LDLQ validation and execution."""


class CBLDLQUnsupportedMode(CBLDLQError):
    """No exact assignment rule is implemented for this serialized mode."""


class CBLDLQHessianError(CBLDLQError):
    """The activation Hessian could not produce a valid feedback factor."""


@dataclass(frozen=True)
class ProductSpec:
    grid: str
    atom_size: int
    subtables_per_vector: int


@dataclass(frozen=True)
class PreparedCBHessian:
    upper_inverse_cholesky: torch.Tensor
    dead_mask: torch.Tensor
    damping: float


@dataclass(frozen=True)
class ProductLDLQResult:
    reconstructed: torch.Tensor
    indices: torch.Tensor
    local_costs: torch.Tensor


def product_spec(*, grid: str, mode: str) -> ProductSpec:
    """Return the physical product atom, failing closed for other modes."""
    if str(mode).strip().lower() != "product":
        raise CBLDLQUnsupportedMode(
            f"CB LDLQ supports only product mode, got {mode!r}; "
            "signed signs are coupled under the dense metric"
        )
    normalized = str(grid).strip().lower()
    if normalized == "fp4":
        return ProductSpec(grid="fp4", atom_size=4, subtables_per_vector=2)
    if normalized == "fp8":
        return ProductSpec(grid="fp8", atom_size=2, subtables_per_vector=4)
    raise CBLDLQUnsupportedMode(
        f"CB LDLQ product grid must be fp4 or fp8, got {grid!r}"
    )


def _compute_dtype(value: torch.Tensor) -> torch.dtype:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def resolve_candidate_workspace_bytes(
    environ: Mapping[str, str] | None = None,
) -> int:
    values = os.environ if environ is None else environ
    raw = str(
        values.get(
            LDLQ_CANDIDATE_WORKSPACE_ENV,
            LDLQ_CANDIDATE_WORKSPACE_DEFAULT,
        )
    ).strip()
    try:
        result = int(raw)
    except ValueError as exc:
        raise CBLDLQError(
            f"{LDLQ_CANDIDATE_WORKSPACE_ENV} must be a positive integer"
        ) from exc
    if result <= 0:
        raise CBLDLQError(
            f"{LDLQ_CANDIDATE_WORKSPACE_ENV} must be a positive integer"
        )
    return result


def candidate_chunk_size(
    *,
    row_instances: int,
    atom_size: int,
    codebook_entries: int,
    element_size: int,
    workspace_bytes: int,
    chunk_cap: int = LDLQ_CANDIDATE_CHUNK_CAP,
    candidate_planes: int = 4,
) -> int:
    """Bound the exhaustive-search temporaries independently of K.

    The conservative peak model counts four atom-width floating planes
    (multiply/candidate, residual/RHS, whitened solve, and overlap) plus the
    scalar cost plane.  The actual eager lifetime is usually smaller, but this
    bound keeps K24 NVFP4 tables from materializing a full ``E*R*4096*4``
    candidate tensor.
    """
    rows = int(row_instances)
    atom = int(atom_size)
    entries = int(codebook_entries)
    scalar_bytes = int(element_size)
    budget = int(workspace_bytes)
    cap = int(chunk_cap)
    if min(rows, atom, entries, scalar_bytes, budget, cap) <= 0:
        raise CBLDLQError("candidate workspace dimensions must be positive")
    planes = int(candidate_planes)
    if planes <= 0:
        raise CBLDLQError("candidate_planes must be positive")
    bytes_per_candidate = rows * (planes * atom + 1) * scalar_bytes
    budget_chunk = max(1, budget // bytes_per_candidate)
    return min(entries, cap, budget_chunk)


def candidate_workspace_bound_bytes(
    *,
    row_instances: int,
    atom_size: int,
    candidate_chunk: int,
    element_size: int,
) -> int:
    """Return the same conservative byte model used to choose a chunk."""
    return (
        int(row_instances)
        * (4 * int(atom_size) + 1)
        * int(candidate_chunk)
        * int(element_size)
    )


def prepare_upper_inverse_cholesky(
    activation_rows: torch.Tensor,
    *,
    device: torch.device,
    damping_fraction: float,
) -> PreparedCBHessian:
    """Build ``U``, failing closed when calibration has dead channels.

    A product-codebook index couples two or four coordinates.  Giving an
    unobserved coordinate an identity-like Hessian diagonal would let its
    artificial residual steer the shared codeword; masking it could still
    change that coordinate without evidence.  Therefore any dead diagonal is
    a typed Hessian failure and the fields layer retains the exact raw index.
    Other malformed/non-finite/factorization failures follow the same path.
    """
    source = torch.as_tensor(activation_rows)
    if source.ndim != 2 or int(source.shape[0]) == 0 or int(source.shape[1]) == 0:
        raise CBLDLQHessianError(
            "activation rows must be a non-empty rank-2 tensor"
        )
    if not bool(torch.isfinite(source).all()):
        raise CBLDLQHessianError("activation rows contain non-finite values")
    if not torch.isfinite(torch.tensor(float(damping_fraction))):
        raise CBLDLQHessianError("damping fraction must be finite")
    if float(damping_fraction) < 0:
        raise CBLDLQHessianError("damping fraction must be non-negative")

    dtype = _compute_dtype(source)
    x = source.to(device=device, dtype=dtype)
    hessian = x.T @ x
    if not bool(torch.isfinite(hessian).all()):
        raise CBLDLQHessianError("activation Hessian contains non-finite values")
    diagonal = hessian.diagonal().clone()
    dead = diagonal <= 0
    if bool(dead.any()):
        raise CBLDLQHessianError(
            "activation Hessian has dead channels; a coupled product "
            "codeword cannot safely reassign unobserved coordinates"
        )
    live_mean = diagonal.mean().clamp_min(torch.finfo(dtype).tiny)
    damping = float(damping_fraction) * float(live_mean)
    hessian.diagonal().add_(damping)

    lower, info = torch.linalg.cholesky_ex(hessian)
    if int(info.max().item()) != 0:
        raise CBLDLQHessianError(
            "damped activation Hessian is not positive definite"
        )
    inverse = torch.cholesky_inverse(lower)
    upper, inverse_info = torch.linalg.cholesky_ex(inverse, upper=True)
    if int(inverse_info.max().item()) != 0 or not bool(torch.isfinite(upper).all()):
        raise CBLDLQHessianError("inverse-Hessian Cholesky failed")
    return PreparedCBHessian(
        upper_inverse_cholesky=upper,
        dead_mask=dead,
        damping=damping,
    )


def _validate_upper(upper: torch.Tensor, columns: int) -> None:
    if upper.shape != (columns, columns):
        raise CBLDLQError(
            f"upper factor shape {tuple(upper.shape)} != {(columns, columns)}"
        )
    if not bool(torch.isfinite(upper).all()) or bool((upper.diagonal() <= 0).any()):
        raise CBLDLQError("upper factor must be finite with a positive diagonal")


def _validate_geometry(
    weight: torch.Tensor,
    *,
    spec: ProductSpec,
    outer_tile_columns: int,
) -> tuple[int, int, int]:
    if weight.ndim != 2:
        raise CBLDLQError(
            f"product LDLQ expects a rank-2 weight, got {tuple(weight.shape)}"
        )
    rows, columns = map(int, weight.shape)
    outer = int(outer_tile_columns)
    if min(rows, columns, outer) <= 0:
        raise CBLDLQError("weight and outer tile must be non-empty/positive")
    if columns % VEC_DIM:
        raise CBLDLQError(
            f"product LDLQ input width {columns} is not vector-{VEC_DIM} aligned"
        )
    if spec.grid == "fp4" and columns % FP4_GROUP:
        raise CBLDLQError(
            f"fp4 product LDLQ input width {columns} is not group-{FP4_GROUP} aligned"
        )
    if outer % spec.atom_size:
        raise CBLDLQError(
            f"outer tile {outer} is not a multiple of atom {spec.atom_size}"
        )
    return rows, columns, outer


def _expand_scales_2d(
    scales: torch.Tensor,
    *,
    rows: int,
    columns: int,
    spec: ProductSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    source = torch.as_tensor(scales, device=device, dtype=dtype)
    if source.ndim == 1:
        source = source.unsqueeze(-1)
    if source.ndim != 2 or int(source.shape[0]) not in (1, rows):
        raise CBLDLQError(
            f"scale shape {tuple(source.shape)} cannot broadcast to {(rows, columns)}"
        )
    if int(source.shape[0]) == 1:
        source = source.expand(rows, source.shape[1])
    if int(source.shape[1]) == columns:
        expanded = source
    elif spec.grid == "fp4" and int(source.shape[1]) == columns // FP4_GROUP:
        expanded = source.repeat_interleave(FP4_GROUP, dim=1)
    elif spec.grid == "fp8" and int(source.shape[1]) == 1:
        expanded = source.expand(rows, columns)
    else:
        raise CBLDLQError(
            f"invalid {spec.grid} scale shape {tuple(source.shape)} for "
            f"weight {(rows, columns)}"
        )
    if not bool(torch.isfinite(expanded).all()) or bool((expanded <= 0).any()):
        raise CBLDLQError("fixed scales must be finite and strictly positive")
    return expanded


def _expand_scales_3d(
    scales: torch.Tensor,
    *,
    experts: int,
    rows: int,
    columns: int,
    spec: ProductSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    source = torch.as_tensor(scales, device=device, dtype=dtype)
    if source.ndim == 2 and int(source.shape[0]) == experts * rows:
        source = source.reshape(experts, rows, source.shape[1])
    if source.ndim != 3 or tuple(source.shape[:2]) != (experts, rows):
        raise CBLDLQError(
            f"3-D scales must be {(experts, rows, 'groups')}, got {tuple(source.shape)}"
        )
    flattened = _expand_scales_2d(
        source.reshape(experts * rows, source.shape[-1]),
        rows=experts * rows,
        columns=columns,
        spec=spec,
        device=device,
        dtype=dtype,
    )
    return flattened.reshape(experts, rows, columns)


def _validate_codebooks(
    codebooks: Sequence[torch.Tensor],
    *,
    spec: ProductSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    tables = tuple(
        torch.as_tensor(table, device=device, dtype=dtype)
        for table in codebooks
    )
    if len(tables) != spec.subtables_per_vector:
        raise CBLDLQError(
            f"{spec.grid} product mode requires {spec.subtables_per_vector} "
            f"subtables, got {len(tables)}"
        )
    for table_index, table in enumerate(tables):
        if (
            table.ndim != 2
            or int(table.shape[0]) == 0
            or int(table.shape[1]) != spec.atom_size
        ):
            raise CBLDLQError(
                f"subtable {table_index} must have shape (K,{spec.atom_size}), "
                f"got {tuple(table.shape)}"
            )
        if not bool(torch.isfinite(table).all()):
            raise CBLDLQError(f"subtable {table_index} contains non-finite values")
    return tables


def _forward_substitute_product_atom(
    residual: torch.Tensor,
    upper_atom: torch.Tensor,
) -> torch.Tensor:
    """Return ``residual @ inverse(upper_atom)`` for a 2-D/4-D atom.

    ``torch.linalg.solve_triangular`` can dispatch a catastrophically slow
    CUDA kernel for these tiny systems when the flattened right-hand side is
    wide.  This fixed-order substitution is mathematically identical to
    solving ``upper_atom.T @ z.T = residual.T`` and avoids constructing that
    RHS.  Packed E16 assignment deliberately keeps the faster batched
    triangular kernel; this helper is the canonical 2-D route only.
    """
    if residual.ndim < 2:
        raise CBLDLQError(
            f"atom residual must have rank >= 2, got {tuple(residual.shape)}"
        )
    atom = int(residual.shape[-1])
    if atom not in (2, 4):
        raise CBLDLQError(
            f"explicit product-atom solve requires width 2 or 4, got {atom}"
        )
    if upper_atom.ndim != 2 or upper_atom.shape != (atom, atom):
        raise CBLDLQError(
            f"2-D atom upper shape {tuple(upper_atom.shape)} != {(atom, atom)}"
        )

    solved: list[torch.Tensor] = []
    for column in range(atom):
        value = residual[..., column]
        # Preserve this explicit left-to-right order.  A reduction/einsum can
        # reintroduce a size-sensitive kernel in the hot path this avoids.
        for prior in range(column):
            value = value - solved[prior] * upper_atom[prior, column]
        value = value / upper_atom[column, column]
        solved.append(value)
    return torch.stack(solved, dim=-1)



# --- compile gate for the LDLQ assignment atoms -------------------------------
# Profiling (2026-08-10, 47,923 samples) put ~88% of DSv4 export runtime under
# ldlq_reassign_cb_fields, with the atom assignment materialising a
# (rows, chunk, atom) candidate residual per chunk. nvfp4_cb_formats already
# compiles its moment-scoring kernels; this file was never compiled at all.
# dynamic=True on purpose: `rows` varies per tensor, and a static compile would
# recompile per shape and blow dynamo's recompile limit, which silently falls
# back to EAGER (the same ~30x trap _raise_encode_recompile_limit documents).
_ATOM_COMPILE_ENV = "PRISMAQUANT_CB_ATOM_COMPILE"
_ATOM_COMPILED = {}


def _atom_compile_on() -> bool:
    import os
    return os.environ.get(_ATOM_COMPILE_ENV, "0").lower() not in ("0", "false", "no")


def _atom_compiled(fn):
    """Compile `fn` once, lazily, when the gate is on; else return it unchanged."""
    if not _atom_compile_on():
        return fn
    strict = cb_compile_fail_closed()
    key = (fn, strict)
    got = _ATOM_COMPILED.get(key)
    if got is None:
        import torch as _t
        try:
            _t._dynamo.config.cache_size_limit = max(
                getattr(_t._dynamo.config, "cache_size_limit", 8), 64)
        except Exception as exc:
            if strict:
                raise CBCompileContractError(
                    "strict CB atom compile could not raise the Dynamo cache limit"
                ) from exc
        if strict:
            helper = {
                "_atom_chunk_best": ATOM_CHUNK_BEST,
                "_batched_chunk_best": ATOM_BATCHED_CHUNK_BEST,
            }.get(getattr(fn, "__name__", ""))
            if helper is None:
                raise CBCompileContractError(
                    f"strict CB atom compile does not recognize {fn!r}"
                )
            got = compile_cb_callable(fn, helper=helper, dynamic=True)
        else:
            got = _t.compile(fn, dynamic=True)
        _ATOM_COMPILED[key] = got
    return got



def _atom_chunk_costs(target, per_element_scale, codebook_chunk, upper_atom):
    """Per-chunk candidate cost. Pure tensor math, no host sync -> fuses whole.

    Kept byte-order-identical to the inline form it replaces: the forward
    substitution retains its explicit left-to-right column order.
    """
    residual = (
        target[:, None, :]
        - per_element_scale[:, None, :] * codebook_chunk[None, :, :]
    )
    whitened = _forward_substitute_product_atom(residual, upper_atom)
    return whitened.square().sum(dim=-1)


def _atom_chunk_best(target, per_element_scale, codebook_chunk, upper_atom):
    """Fused residual -> substitute -> squared-norm -> argmin.

    The reduction MUST happen inside the compiled region.  Returning the full
    (rows, cand) cost tensor made inductor materialise tens of MB per atom
    step and the path became memcpy-bound -- 65% of runtime sat in
    Device->Device copies while the arithmetic kernel was 6%.  Reducing
    in-kernel writes (rows,) instead of (rows, cand).

    Squares accumulate per column rather than via ``torch.stack``: the stack
    forces a real (rows, cand, atom) buffer that defeats the fusion.
    """
    residual = (
        target[:, None, :]
        - per_element_scale[:, None, :] * codebook_chunk[None, :, :]
    )
    atom = residual.shape[-1]
    solved = []
    total = None
    for column in range(atom):
        value = residual[..., column]
        for prior in range(column):
            value = value - solved[prior] * upper_atom[prior, column]
        value = value / upper_atom[column, column]
        solved.append(value)
        total = value.square() if total is None else total + value.square()
    return total.min(dim=-1)


def _atom_chunk_costs_batched(target, per_element_scale, codebook_chunk,
                              upper_atom):
    residual = (
        target[..., None, :]
        - per_element_scale[..., None, :] * codebook_chunk[None, None, :, :]
    )
    whitened = _forward_substitute_product_atom(residual, upper_atom)
    return whitened.square().sum(dim=-1)


def _fused_route(target: torch.Tensor) -> bool:
    """Whether the fused compiled route may run.  CUDA ONLY -- deliberately.

    torch.compile's CPU lowering of ``min(dim=-1)`` returns the correct
    minimum VALUE paired with a WRONG index.  Measured on this box
    (torch 2.11+cu130): 381/512 rows wrong at fp32 and 396/512 at fp64, the
    returned index costing up to 31x the true minimum, while CUDA is exact
    (0/512 wrong, zero excess cost) for both dtypes.

    The index is precisely what gets serialised into the artifact, and the
    returned cost still looks right, so a CPU fused route would corrupt an
    export silently and pass any check that only inspects costs.  CPU
    therefore always takes the eager route.  This costs nothing in
    production: every hot path here is GPU-or-bust by design.
    """
    enabled = _atom_compile_on()
    if enabled and cb_compile_fail_closed() and not bool(target.is_cuda):
        refuse_cb_compile_fallback(
            ATOM_BATCHED_CHUNK_BEST if target.ndim == 3 else ATOM_CHUNK_BEST,
            reason="index-producing CB atom compile is CUDA-only",
        )
    return enabled and bool(target.is_cuda)


def _fused_candidate_planes(target: torch.Tensor) -> int:
    """Live candidate planes under the active route.

    The eager route really does hold four atom-width planes at once
    (candidate product, residual, whitened solve, overlap).  The fused route
    reduces to (rows,) inside one Triton kernel, so only the residual plane
    is ever live.  Keeping the eager number under compilation split every
    atom step into three ragged chunks (113 of 256), tripling compiled-graph
    launches for no memory benefit.
    """
    return 1 if _fused_route(target) else 4


def assign_product_atom(
    target: torch.Tensor,
    per_element_scale: torch.Tensor,
    codebook: torch.Tensor,
    upper_atom: torch.Tensor,
    *,
    workspace_bytes: int,
    chunk_cap: int = LDLQ_CANDIDATE_CHUNK_CAP,
    defer_finite_check: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact exhaustive Mahalanobis assignment with bounded workspace.

    ``defer_finite_check`` moves the non-finite guard to the caller.  The
    guard is a host synchronisation; running it once per atom step drained
    the pipeline columns/atom times per matrix (400 ms of 608 ms in a
    profile).  ``reassign_product_*`` re-check the identical condition on
    the stacked costs they already build, so the failure is still closed --
    only later.
    """
    rows, atom = map(int, target.shape)
    entries = int(codebook.shape[0])
    chunk = candidate_chunk_size(
        row_instances=rows,
        atom_size=atom,
        codebook_entries=entries,
        element_size=target.element_size(),
        workspace_bytes=workspace_bytes,
        chunk_cap=chunk_cap,
        candidate_planes=_fused_candidate_planes(target),
    )
    best_cost = torch.full(
        (rows,), float("inf"), device=target.device, dtype=target.dtype
    )
    best_index = torch.zeros((rows,), device=target.device, dtype=torch.long)
    for first in range(0, entries, chunk):
        last = min(first + chunk, entries)
        if _fused_route(target):
            local_cost, local_index = _atom_compiled(_atom_chunk_best)(
                target, per_element_scale, codebook[first:last], upper_atom
            )
        else:
            costs = _atom_chunk_costs(
                target, per_element_scale, codebook[first:last], upper_atom
            )
            local_cost, local_index = costs.min(dim=-1)
        improve = local_cost < best_cost
        best_cost = torch.where(improve, local_cost, best_cost)
        best_index = torch.where(improve, local_index + first, best_index)
    if not defer_finite_check and not bool(torch.isfinite(best_cost).all()):
        raise CBLDLQError("atom assignment produced a non-finite cost")
    decoded = per_element_scale * codebook[best_index]
    return decoded, best_index, best_cost



def _batched_rhs(target, per_element_scale, codebook_chunk,
                 experts, atom, rows, n_cand):
    """Residual + layout for the batched triangular solve (fuses; no sync)."""
    residual = (
        target[:, :, None, :]
        - per_element_scale[:, :, None, :] * codebook_chunk[None, None, :, :]
    )
    return residual.permute(0, 3, 1, 2).reshape(experts, atom, rows * n_cand)


def _batched_costs(whitened, experts, atom, rows, n_cand):
    """Post-solve reshape + squared-norm reduction (fuses; no sync)."""
    w = whitened.reshape(experts, atom, rows, n_cand).permute(0, 2, 3, 1)
    return w.square().sum(dim=-1)



def _forward_substitute_batched(residual, upper_atom):
    """Per-expert forward substitution, same recurrence as the 2-D atom path.

    Replaces torch.linalg.solve_triangular for the batched route. cuSOLVER on a
    4x4 (fp4) / 2x2 (fp8) system is a library call the compiler cannot fuse
    through, which pinned the batched path at ~1.25x while the 2-D path reached
    ~5.9x. This is the identical mathematical recurrence, expressed in ops that
    fuse into the surrounding residual/square/sum chain.

    residual: (E, rows, cand, atom)   upper_atom: (E, atom, atom)
    Solves U^T y = residual, matching solve_triangular(U^T, ., upper=False).
    """
    atom = residual.shape[-1]
    solved = []
    for column in range(atom):
        value = residual[..., column]
        for prior in range(column):
            value = value - solved[prior] * upper_atom[:, prior, column].reshape(-1, 1, 1)
        value = value / upper_atom[:, column, column].reshape(-1, 1, 1)
        solved.append(value)
    return torch.stack(solved, dim=-1)


def _batched_chunk_best(target, per_element_scale, codebook_chunk, upper_atom):
    """Per-expert fused residual -> substitute -> squared-norm -> argmin.

    See _atom_chunk_best: the reduction must live inside the compiled region
    or the (E, rows, cand) cost tensor dominates as Device->Device traffic.
    """
    residual = (
        target[:, :, None, :]
        - per_element_scale[:, :, None, :] * codebook_chunk[None, None, :, :]
    )
    atom = residual.shape[-1]
    solved = []
    total = None
    for column in range(atom):
        value = residual[..., column]
        for prior in range(column):
            value = value - solved[prior] * upper_atom[:, prior, column].reshape(-1, 1, 1)
        value = value / upper_atom[:, column, column].reshape(-1, 1, 1)
        solved.append(value)
        total = value.square() if total is None else total + value.square()
    return total.min(dim=-1)


def _batched_whitened_costs(target, per_element_scale, codebook_chunk, upper_atom):
    """Fused residual -> forward-substitute -> squared-norm. No sync, no library call."""
    residual = (
        target[:, :, None, :]
        - per_element_scale[:, :, None, :] * codebook_chunk[None, None, :, :]
    )
    whitened = _forward_substitute_batched(residual, upper_atom)
    return whitened.square().sum(dim=-1)


def assign_product_atom_batched(
    target: torch.Tensor,
    per_element_scale: torch.Tensor,
    codebook: torch.Tensor,
    upper_atom: torch.Tensor,
    *,
    workspace_bytes: int,
    chunk_cap: int = LDLQ_CANDIDATE_CHUNK_CAP,
    defer_finite_check: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert exact atom assignment, vectorized across experts/rows.

    See ``assign_product_atom`` for ``defer_finite_check``.
    """
    experts, rows, atom = map(int, target.shape)
    entries = int(codebook.shape[0])
    chunk = candidate_chunk_size(
        row_instances=experts * rows,
        atom_size=atom,
        codebook_entries=entries,
        element_size=target.element_size(),
        workspace_bytes=workspace_bytes,
        chunk_cap=chunk_cap,
        candidate_planes=_fused_candidate_planes(target),
    )
    best_cost = torch.full(
        (experts, rows),
        float("inf"),
        device=target.device,
        dtype=target.dtype,
    )
    best_index = torch.zeros(
        (experts, rows), device=target.device, dtype=torch.long
    )
    for first in range(0, entries, chunk):
        last = min(first + chunk, entries)
        if _fused_route(target):
            # Fused route: explicit substitution + argmin compile into a single
            # reduction kernel. cuSOLVER cannot be fused through, and the cost
            # tensor is never materialised.
            local_cost, local_index = _atom_compiled(_batched_chunk_best)(
                target, per_element_scale, codebook[first:last], upper_atom
            )
        else:
            # Default route, unchanged: eager forward substitution is SLOWER
            # than cuSOLVER, so the gate-off path must keep the library call.
            residual = (
                target[:, :, None, :]
                - per_element_scale[:, :, None, :]
                * codebook[None, None, first:last, :]
            )
            rhs = residual.permute(0, 3, 1, 2).reshape(
                experts, atom, rows * (last - first)
            )
            whitened = torch.linalg.solve_triangular(
                upper_atom.transpose(-2, -1),
                rhs,
                upper=False,
            ).reshape(experts, atom, rows, last - first).permute(0, 2, 3, 1)
            costs = whitened.square().sum(dim=-1)
            local_cost, local_index = costs.min(dim=-1)
        improve = local_cost < best_cost
        best_cost = torch.where(improve, local_cost, best_cost)
        best_index = torch.where(improve, local_index + first, best_index)
    if not defer_finite_check and not bool(torch.isfinite(best_cost).all()):
        raise CBLDLQError("batched atom assignment produced a non-finite cost")
    decoded = per_element_scale * codebook[best_index]
    return decoded, best_index, best_cost


def reassign_product_2d(
    weight: torch.Tensor,
    scales: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    upper_inverse_cholesky: torch.Tensor,
    *,
    grid: str,
    mode: str,
    outer_tile_columns: int = LDLQ_OUTER_TILE_COLUMNS,
    candidate_workspace_bytes: int | None = None,
) -> ProductLDLQResult:
    """Reassign fixed product indices for one matrix; scales/books stay fixed."""
    spec = product_spec(grid=grid, mode=mode)
    source = torch.as_tensor(weight)
    rows, columns, outer = _validate_geometry(
        source, spec=spec, outer_tile_columns=outer_tile_columns
    )
    dtype = _compute_dtype(source)
    work = source.to(dtype=dtype).clone()
    upper = torch.as_tensor(
        upper_inverse_cholesky, device=source.device, dtype=dtype
    )
    _validate_upper(upper, columns)
    expanded_scales = _expand_scales_2d(
        scales,
        rows=rows,
        columns=columns,
        spec=spec,
        device=source.device,
        dtype=dtype,
    )
    tables = _validate_codebooks(
        codebooks, spec=spec, device=source.device, dtype=dtype
    )
    workspace = (
        resolve_candidate_workspace_bytes()
        if candidate_workspace_bytes is None
        else int(candidate_workspace_bytes)
    )
    if workspace <= 0:
        raise CBLDLQError("candidate_workspace_bytes must be positive")

    reconstructed = torch.empty_like(work)
    assignment_parts: list[torch.Tensor] = []
    cost_parts: list[torch.Tensor] = []
    atom = spec.atom_size
    for tile_start in range(0, columns, outer):
        tile_end = min(tile_start + outer, columns)
        tile = work[:, tile_start:tile_end].clone()
        solved_errors = torch.zeros_like(tile)
        for start in range(tile_start, tile_end, atom):
            end = start + atom
            local_start = start - tile_start
            local_end = end - tile_start
            upper_atom = upper[start:end, start:end]
            table = tables[(start % VEC_DIM) // atom]
            decoded, index, local_cost = assign_product_atom(
                tile[:, local_start:local_end],
                expanded_scales[:, start:end],
                table,
                upper_atom,
                workspace_bytes=workspace,
                defer_finite_check=True,
            )
            residual = tile[:, local_start:local_end] - decoded
            scaled_error = _forward_substitute_product_atom(
                residual, upper_atom
            )
            reconstructed[:, start:end] = decoded
            assignment_parts.append(index)
            cost_parts.append(local_cost)
            solved_errors[:, local_start:local_end] = scaled_error
            if local_end < tile_end - tile_start:
                tile[:, local_end:] -= (
                    scaled_error @ upper[start:end, end:tile_end]
                )
        if tile_end < columns:
            work[:, tile_end:] -= (
                solved_errors @ upper[tile_start:tile_end, tile_end:]
            )

    vectors = columns // VEC_DIM
    stacked_costs = torch.stack(cost_parts, dim=1)
    if not bool(torch.isfinite(stacked_costs).all()):
        raise CBLDLQError("atom assignment produced a non-finite cost")
    return ProductLDLQResult(
        reconstructed=reconstructed,
        indices=torch.stack(assignment_parts, dim=1).reshape(
            rows, vectors, spec.subtables_per_vector
        ),
        local_costs=stacked_costs.reshape(
            rows, vectors, spec.subtables_per_vector
        ),
    )


def reassign_product_3d_batched(
    weight: torch.Tensor,
    scales: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    upper_inverse_cholesky: torch.Tensor,
    *,
    grid: str,
    mode: str,
    outer_tile_columns: int = LDLQ_OUTER_TILE_COLUMNS,
    candidate_workspace_bytes: int | None = None,
) -> ProductLDLQResult:
    """Correct per-expert metric with assignment/feedback batched over E."""
    spec = product_spec(grid=grid, mode=mode)
    source = torch.as_tensor(weight)
    if source.ndim != 3:
        raise CBLDLQError(
            f"batched product LDLQ expects [E,R,C], got {tuple(source.shape)}"
        )
    experts, rows, columns = map(int, source.shape)
    _validate_geometry(
        source[0], spec=spec, outer_tile_columns=outer_tile_columns
    )
    dtype = _compute_dtype(source)
    work = source.to(dtype=dtype).clone()
    upper = torch.as_tensor(
        upper_inverse_cholesky, device=source.device, dtype=dtype
    )
    if upper.shape != (experts, columns, columns):
        raise CBLDLQError(
            f"batched upper shape {tuple(upper.shape)} != "
            f"{(experts, columns, columns)}"
        )
    for expert in range(experts):
        _validate_upper(upper[expert], columns)
    expanded_scales = _expand_scales_3d(
        scales,
        experts=experts,
        rows=rows,
        columns=columns,
        spec=spec,
        device=source.device,
        dtype=dtype,
    )
    tables = _validate_codebooks(
        codebooks, spec=spec, device=source.device, dtype=dtype
    )
    workspace = (
        resolve_candidate_workspace_bytes()
        if candidate_workspace_bytes is None
        else int(candidate_workspace_bytes)
    )
    if workspace <= 0:
        raise CBLDLQError("candidate_workspace_bytes must be positive")

    reconstructed = torch.empty_like(work)
    assignment_parts: list[torch.Tensor] = []
    cost_parts: list[torch.Tensor] = []
    atom = spec.atom_size
    outer = int(outer_tile_columns)
    for tile_start in range(0, columns, outer):
        tile_end = min(tile_start + outer, columns)
        tile = work[:, :, tile_start:tile_end].clone()
        solved_errors = torch.zeros_like(tile)
        for start in range(tile_start, tile_end, atom):
            end = start + atom
            local_start = start - tile_start
            local_end = end - tile_start
            upper_atom = upper[:, start:end, start:end]
            table = tables[(start % VEC_DIM) // atom]
            decoded, index, local_cost = assign_product_atom_batched(
                tile[:, :, local_start:local_end],
                expanded_scales[:, :, start:end],
                table,
                upper_atom,
                workspace_bytes=workspace,
                defer_finite_check=True,
            )
            residual = tile[:, :, local_start:local_end] - decoded
            scaled_error = torch.linalg.solve_triangular(
                upper_atom.transpose(-2, -1),
                residual.transpose(-2, -1),
                upper=False,
            ).transpose(-2, -1)
            reconstructed[:, :, start:end] = decoded
            assignment_parts.append(index)
            cost_parts.append(local_cost)
            solved_errors[:, :, local_start:local_end] = scaled_error
            if local_end < tile_end - tile_start:
                tile[:, :, local_end:] -= torch.bmm(
                    scaled_error,
                    upper[:, start:end, end:tile_end],
                )
        if tile_end < columns:
            work[:, :, tile_end:] -= torch.bmm(
                solved_errors,
                upper[:, tile_start:tile_end, tile_end:],
            )

    vectors = columns // VEC_DIM
    stacked_costs = torch.stack(cost_parts, dim=2)
    if not bool(torch.isfinite(stacked_costs).all()):
        raise CBLDLQError("batched atom assignment produced a non-finite cost")
    return ProductLDLQResult(
        reconstructed=reconstructed,
        indices=torch.stack(assignment_parts, dim=2).reshape(
            experts, rows, vectors, spec.subtables_per_vector
        ),
        local_costs=stacked_costs.reshape(
            experts, rows, vectors, spec.subtables_per_vector
        ),
    )


__all__ = [
    "CBLDLQError",
    "CBLDLQHessianError",
    "CBLDLQUnsupportedMode",
    "LDLQ_CANDIDATE_WORKSPACE_DEFAULT",
    "LDLQ_CANDIDATE_WORKSPACE_ENV",
    "LDLQ_OUTER_TILE_COLUMNS",
    "PreparedCBHessian",
    "ProductLDLQResult",
    "assign_product_atom",
    "assign_product_atom_batched",
    "candidate_chunk_size",
    "candidate_workspace_bound_bytes",
    "prepare_upper_inverse_cholesky",
    "product_spec",
    "reassign_product_2d",
    "reassign_product_3d_batched",
    "resolve_candidate_workspace_bytes",
]
