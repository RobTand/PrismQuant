"""Randomized Hadamard rotation for the rotation-control study.

Implements Sylvester Hadamard + Rademacher diagonal (H·D) on the input dimension
(the dimension the product codebook vectorizes over, VEC_DIM=8 in
prismaquant/nvfp4_cb_formats.py:69).  Dimensions are powers of two (4096/2048)
so plain Sylvester construction works.

Contract (critical correctness):
  quantize in rotated space, reconstruct, UNROTATE back, compute weighted MSE
  in original space against original weights with original col_weights.

See also prismaquant/rotation_ldlq_pilot.py:64-112 (reference fast FWHT).
"""
from __future__ import annotations

import torch


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def build_hadamard(n: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return normalized Sylvester Hadamard (n,n) with H @ H.T = I (up to fp32).
    
    Normalized by 1/sqrt(n).  Only power-of-two n supported, matching study
    geometry (4096/2048).  Reference: prismaquant/nvfp4_cb_formats.py VEC_DIM=8
    vectorization over input columns.
    """
    if not is_power_of_two(n):
        raise ValueError(f"Hadamard requires power-of-two n, got {n}")
    # Sylvester recursion: H1=[1], H2n = [[Hn,Hn],[Hn,-Hn]]
    H = torch.ones((1, 1), dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = H.to(device=device, dtype=dtype) * (n ** -0.5)
    return H


def random_hadamard_signs(width: int, seed: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return Rademacher ±1 diagonal for randomized Hadamard, shape (width,).
    
    Mirrors prismaquant/rotation_ldlq_pilot.py:64 -- deterministic CPU generator,
    float32 signs for FWHT.  Seed recorded per unit for provenance.
    """
    if not is_power_of_two(width):
        raise ValueError(f"Hadamard width must be power of two, got {width}")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    bits = torch.randint(0, 2, (width,), generator=g, dtype=torch.int64)
    signs = bits.to(torch.float32).mul_(2.0).sub_(1.0)  # 0-> -1, 1->+1
    return signs.to(device=device)


def _fwht_right(tensor: torch.Tensor) -> torch.Tensor:
    """Apply normalized Hadamard H_n / sqrt(n) on the last dimension, no signs.
    
    Fast Walsh-Hadamard Transform butterfly, as in
    prismaquant/rotation_ldlq_pilot.py:80-112.  Works for any leading shape,
    operates on last dim which must be power of two.  Arithmetic matches:
      x @ H_n  (H_n = Sylvester / sqrt(n), symmetric and orthogonal).
    """
    width = int(tensor.shape[-1])
    if not is_power_of_two(width):
        raise ValueError(f"FWHT width must be power of two, got {width}")
    work_dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    work = tensor.to(work_dtype)
    leading_shape = tuple(work.shape[:-1])
    step = 1
    while step < width:
        paired = work.reshape(*leading_shape, -1, 2 * step)
        left = paired[..., :step]
        right = paired[..., step:]
        work = torch.cat((left + right, left - right), dim=-1).reshape(*leading_shape, width)
        step *= 2
    work = work * (width ** -0.5)
    return work.to(tensor.dtype)


def apply_rotation(tensor: torch.Tensor, signs: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Apply randomized Hadamard rotation on last dim.
    
    Forward (inverse=False):  tensor @ (H_n @ diag(signs))  i.e. H·D
      = fwht(tensor) * signs   (H first, then column scaling)
    Inverse (inverse=True):   tensor @ (diag(signs) @ H_n)  i.e. D·H
      = fwht(tensor * signs)  (row scaling, then H)
    
    They are inverses: inverse(forward(x)) = x  because H_n^2 = I and D^2 = I,
    and forward and inverse are transposes (H symmetric).  Spec says H·D,
    so forward is H·D, inverse is D·H = (H·D).T.
    
    Args:
        tensor: (..., width)  weight matrix over input dim
        signs: (width,)  ±1
        inverse: if True apply D·H (the transpose) to unrotate
    """
    width = int(tensor.shape[-1])
    if signs.shape != (width,):
        raise ValueError(f"signs shape {tuple(signs.shape)} != {(width,)}")
    if inverse:
        # D·H  == fwht(x * signs)
        return _fwht_right(tensor * signs.to(tensor.device, tensor.dtype))
    else:
        # H·D  == fwht(x) * signs
        return _fwht_right(tensor) * signs.to(tensor.device, tensor.dtype)


def rotate_weights(weight: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Rotate weight matrix on input (last) dim: W_rot = W @ (H·D)."""
    return apply_rotation(weight, signs, inverse=False)


def unrotate_weights(weight_rot: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Unrotate: W_rec_orig = W_rec_rot @ (D·H) = (H·D).T."""
    return apply_rotation(weight_rot, signs, inverse=True)


def build_rotation_matrix(n: int, signs: torch.Tensor, *, forward: bool = True, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dense rotation matrix for tests (n,n). Not used for large-n encode."""
    H = build_hadamard(n, device=device, dtype=dtype)
    s = signs.to(device=device, dtype=dtype)
    D = torch.diag(s)
    if forward:
        return H @ D  # H·D
    else:
        return D @ H  # D·H
