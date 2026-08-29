"""Shared immutable contracts for sample-parallel activation caches.

Activation priorities are a keyed permutation of the unsigned 32-bit global
sample/token row.  The implementation deliberately uses nonnegative int64
intermediates and an overflow-free 16-bit-limb ``mullo32``: every mathematical
intermediate is below ``2**49``.  CPU and CUDA therefore agree by construction;
the contract never relies on unsigned tensor support, signed overflow, RNG
state, floating-point ordering, or compiler-specific wraparound.
"""
from __future__ import annotations

import hashlib
import re
import struct

import torch

from prismaquant.perturbed_x_cache import fused_subsample_group


ACTIVATION_CACHE_SHARD_SCHEMA = (
    "prismaquant.sample_parallel_probe.activation_cache_shard.v1"
)
ACTIVATION_PRIORITY_SCHEMA = (
    "blake2b64-keyed-fmix32x2-prp-global-row-fused-group-v2"
)
ROW_INDICES_SCOPE = "shard_local_flat_tokens"

ACTIVATION_PRIORITY_KEY_DOMAIN = (
    b"prismaquant.sample-parallel.activation-priority.v2\0"
)
ACTIVATION_PRIORITY_MAX_ROWS = 1 << 32
_U16_MASK = (1 << 16) - 1
_U32_MASK = (1 << 32) - 1
_FMIX32_C1 = 0x85EBCA6B
_FMIX32_C2 = 0xC2B2AE35
_CALIBRATION_HASH_RE = re.compile(r"[0-9a-f]{32}")


def activation_priority_group(qname: str) -> str:
    """Resolve the existing joint-consumer row-alignment group."""
    return fused_subsample_group(str(qname))


def validate_activation_priority_domain(
    global_samples: int,
    seqlen: int,
) -> int:
    """Return the global dense row count or refuse the uint32 PRP domain."""
    samples = int(global_samples)
    tokens = int(seqlen)
    total = samples * tokens
    if (
        samples < 1
        or tokens < 1
        or total < 1
        or total > ACTIVATION_PRIORITY_MAX_ROWS
    ):
        raise ValueError(
            "sample-parallel activation priority domain requires "
            "0 < global_samples * seqlen <= 2**32"
        )
    return total


def activation_priority_key(
    global_calibration_hash: str,
    qname_or_group: str,
    *,
    is_group: bool = False,
) -> tuple[int, int]:
    """Derive the exact little-endian uint32 key pair for one fused group."""
    calibration_hash = str(global_calibration_hash)
    if _CALIBRATION_HASH_RE.fullmatch(calibration_hash) is None:
        raise ValueError("global calibration hash must be 32 lowercase hex digits")
    group = (
        str(qname_or_group)
        if is_group
        else activation_priority_group(str(qname_or_group))
    )
    encoded_group = group.encode("utf-8")
    if len(encoded_group) > _U32_MASK:
        raise ValueError("activation priority group name is too long")
    material = (
        ACTIVATION_PRIORITY_KEY_DOMAIN
        + bytes.fromhex(calibration_hash)
        + struct.pack("<I", len(encoded_group))
        + encoded_group
    )
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return struct.unpack("<II", digest)


def _mullo32_scalar(value: int, constant: int) -> int:
    """Low uint32 product without a signed-overflowing intermediate."""
    x = int(value) & _U32_MASK
    c = int(constant) & _U32_MASK
    low = x & _U16_MASK
    high = x >> 16
    return (
        low * c + ((high * (c & _U16_MASK)) << 16)
    ) & _U32_MASK


def _fmix32_scalar(value: int) -> int:
    x = int(value) & _U32_MASK
    x ^= x >> 16
    x = _mullo32_scalar(x, _FMIX32_C1)
    x ^= x >> 13
    x = _mullo32_scalar(x, _FMIX32_C2)
    x ^= x >> 16
    return x & _U32_MASK


def activation_row_priority_scalar(
    global_calibration_hash: str,
    qname: str,
    global_row: int,
) -> int:
    """Normative arbitrary-precision reference for one global row."""
    row = int(global_row)
    if row < 0 or row >= ACTIVATION_PRIORITY_MAX_ROWS:
        raise ValueError("global activation row is outside the uint32 domain")
    k0, k1 = activation_priority_key(global_calibration_hash, qname)
    return _fmix32_scalar(_fmix32_scalar(row ^ k0) ^ k1)


def _mullo32_tensor(values: torch.Tensor, constant: int) -> torch.Tensor:
    """Vectorized low uint32 product with all int64 intermediates < 2**49."""
    low = torch.bitwise_and(values, _U16_MASK)
    high = torch.bitwise_right_shift(values, 16)
    return torch.bitwise_and(
        low * int(constant)
        + torch.bitwise_left_shift(
            high * int(constant & _U16_MASK), 16,
        ),
        _U32_MASK,
    )


def _fmix32_tensor(values: torch.Tensor) -> torch.Tensor:
    x = torch.bitwise_and(values, _U32_MASK)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = _mullo32_tensor(x, _FMIX32_C1)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 13))
    x = _mullo32_tensor(x, _FMIX32_C2)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    return torch.bitwise_and(x, _U32_MASK)


def activation_row_priorities(
    global_calibration_hash: str,
    qname: str,
    global_rows: torch.Tensor,
) -> torch.Tensor:
    """Return device-preserving, globally comparable uint32 priorities.

    ``global_rows`` must already be within the validated global uint32 domain.
    Producer callers construct those rows from a closed partition contract;
    the independent merger checks every persisted row and priority on CPU.
    """
    if not isinstance(global_rows, torch.Tensor) or global_rows.ndim != 1:
        raise ValueError("global activation rows must be a one-dimensional tensor")
    rows = global_rows.detach().to(dtype=torch.long)
    k0, k1 = activation_priority_key(global_calibration_hash, qname)
    first = _fmix32_tensor(torch.bitwise_xor(rows, k0))
    return _fmix32_tensor(torch.bitwise_xor(first, k1))


__all__ = [
    "ACTIVATION_CACHE_SHARD_SCHEMA",
    "ACTIVATION_PRIORITY_KEY_DOMAIN",
    "ACTIVATION_PRIORITY_MAX_ROWS",
    "ACTIVATION_PRIORITY_SCHEMA",
    "ROW_INDICES_SCOPE",
    "activation_priority_group",
    "activation_priority_key",
    "activation_row_priorities",
    "activation_row_priority_scalar",
    "validate_activation_priority_domain",
]
