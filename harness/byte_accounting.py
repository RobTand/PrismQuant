"""
Byte accounting for FP8_CB within-linear splitting.

Geometry (prismaquant/cb_layout.py : VEC_DIM=8, SUPERBLOCK=256):
- FP8 product n_sub=4, sub_dim=2, VEC_DIM=8
- Packing: 32 codewords per 256-weight superblock, 4*K bytes per superblock
- FP8 row scale: 4 bytes per output row (fp32), shared across all column groups of a Linear.

Split model: contiguous column blocks, each multiple of SUPERBLOCK (256) and VEC_DIM (8).
All candidate group counts 2,4,8 divide both gate/up (4096) and down (2048) exactly.

Bytes are tensor payload bytes (compressed-tensors data span), sidecars are deduped
and amortized <<1% model-wide, excluded from per-linear equal-bytes (noted in ledger).
FP8 row scale is 4*out per Linear.  For split we assume ONE scale per row shared across
groups (one GEMM with two index planes, same scale).  Alternative per-group scale
(G*4*out) is quantified as <0.6% overhead and does not change GO/NO-GO.
"""
from __future__ import annotations

from dataclasses import dataclass

SUPERBLOCK = 256
VEC_DIM = 8
INDEX_BYTES_PER_K = 4  # CODEWORDS_PER_SUPERBLOCK//8 * k = 4*k
FP32_BYTES = 4

# Per-expert Linear shapes for DeepSeek-V4-Flash MoE (from burn-shard identities):
# gate_proj/up_proj: (2048, 4096), down_proj: (4096, 2048)
PER_PROJ_SHAPE = {
    "gate_proj": (2048, 4096),
    "up_proj": (2048, 4096),
    "down_proj": (4096, 2048),
}

def _shape_for(proj: str) -> tuple[int, int]:
    if proj not in PER_PROJ_SHAPE:
        raise ValueError(f"unknown projection {proj!r}")
    return PER_PROJ_SHAPE[proj]

def bytes_uniform(projection: str, k: int, *, per_group_scale: bool = False, groups: int = 1) -> int:
    """Tensor payload bytes for a uniform (or per-group scaled) Linear.

    For FP8_CB: index_bytes = out * (in//256) * 4*k, fp8_row_scale = 4*out
    If per_group_scale=True, scale is G*4*out (one per sub-GEMM).
    """
    out, inn = _shape_for(projection)
    n_sb = inn // SUPERBLOCK
    index_bytes = out * n_sb * INDEX_BYTES_PER_K * int(k)
    if per_group_scale:
        scale_bytes = FP32_BYTES * out * int(groups)
    else:
        scale_bytes = FP32_BYTES * out
    return int(index_bytes + scale_bytes)

def bytes_split(projection: str, ks: list[int], *, per_group_scale: bool = False) -> int:
    """Total bytes for a split assignment ks per column group (len=groups).

    Groups are equal-sized contiguous blocks (in_g = in // G).
    Each group's bytes = out * (in_g//256) * 4*k_g (+ 4*out if per_group_scale)
    Shared-scale model charges 4*out once total.
    """
    out, inn = _shape_for(projection)
    g = len(ks)
    if inn % g != 0:
        raise ValueError(f"in={inn} not divisible by groups={g}")
    in_g = inn // g
    if in_g % SUPERBLOCK != 0:
        raise ValueError(f"group width {in_g} not multiple of {SUPERBLOCK}")
    if in_g % VEC_DIM != 0:
        raise ValueError(f"group width {in_g} not multiple of {VEC_DIM}")
    # superblocks per group-row
    sb_g = in_g // SUPERBLOCK
    index_total = 0
    for k in ks:
        index_total += out * sb_g * INDEX_BYTES_PER_K * int(k)
    if per_group_scale:
        scale_total = FP32_BYTES * out * g
    else:
        scale_total = FP32_BYTES * out
    return int(index_total + scale_total)

def bytes_for_budget(projection: str, k_budget: int, *, per_group_scale: bool = False, groups: int = 1) -> int:
    """Convenience: budget bytes equivalent to uniform K_budget."""
    return bytes_uniform(projection, k_budget, per_group_scale=per_group_scale, groups=groups)

def ledger_row(projection: str, uniform_k: int, split_ks: list[int], *, per_group_scale: bool = False) -> dict:
    """Return auditable ledger showing equality (or delta) between uniform and split."""
    uniform_bytes = bytes_uniform(projection, uniform_k, per_group_scale=per_group_scale, groups=len(split_ks))
    # For uniform bytes reference with shared model, compare against shared total too
    split_bytes = bytes_split(projection, split_ks, per_group_scale=per_group_scale)
    return {
        "projection": projection,
        "uniform_k": uniform_k,
        "split_ks": list(split_ks),
        "groups": len(split_ks),
        "per_group_scale": bool(per_group_scale),
        "uniform_bytes": uniform_bytes,
        "split_bytes": split_bytes,
        "delta_bytes": split_bytes - uniform_bytes,
        "equal": split_bytes == uniform_bytes,
        # breakdown for hand check
        "out_in": _shape_for(projection),
        "superblocks_per_group": _shape_for(projection)[1] // len(split_ks) // SUPERBLOCK,
    }

def bpp(tensor_payload_bytes: int, projection: str) -> float:
    """Bits per parameter for one Linear (quantizable params only)."""
    out, inn = _shape_for(projection)
    params = out * inn
    return 8.0 * tensor_payload_bytes / params

def uniform_bpp(projection: str, k: int) -> float:
    return bpp(bytes_uniform(projection, k), projection)

# Example ledgers for acceptance criteria (3 hand-checked, shared-scale model exact equality)
EXAMPLE_LEDGERS = [
    ledger_row("gate_proj", 32, [30, 34]),
    ledger_row("gate_proj", 33, [31, 33, 33, 35]),
    ledger_row("down_proj", 36, [34, 34, 38, 38]),
]
