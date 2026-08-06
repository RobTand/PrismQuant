"""Rotation-aware encode wrappers for incumbent vs CBL.

Protocol (critical):
  - rotate weight on input dim: W_rot = W @ (H·D)
  - quantize W_rot in rotated space (incumbent fixed-lattice or CBL pool)
  - reconstruct R_rot
  - unrotate: R_orig = R_rot @ (D·H)  == R_rot @ (H·D).T
  - score weighted MSE in ORIGINAL space vs original W with original col_weights

This file provides helpers that wrap the real encoders in
tools/dsv4_cbl_kernels.py: encode_free_noldlq (incumbent) and encode_cbl (CBL)
as well as a synthetic fallback using prismaquant/nvfp4_cb_formats.py direct.
"""
from __future__ import annotations

import torch

from .rotation import rotate_weights, unrotate_weights


def quantize_with_rotation(
    weight: torch.Tensor,
    col_weights: torch.Tensor,
    signs: torch.Tensor,
    quantize_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper: rotate, quantize via quantize_fn, unrotate.
    
    quantize_fn: callable(W_rot, cw_rot?) -> R_rot  (reconstruction in rotated space)
    For incumbent weighted case we preserve original col_weights semantics:
      - If col_weights are provided, we pass col_weights_rot = col_weights *?? ?
      For this control we KEEP original col_weights values indexed by rotated
      column position (i.e., no transformation of weights).  The alternative
      would be to also rotate col_weights, but that would be inconsistent with
      the spec's "compute weighted MSE in original space with original
      col_weights" evaluation — the encoder's internal weighting is a separate
      choice.  We document both and test identity case.
      Here we pass original col_weights unchanged to the quantizer operating
      on rotated weights (uniform-ish after Hadamard).  For a uniform
      Hadamard, the per-column importance is equalized, so weighting matters
      less, but we keep the same weighting for strict comparability.
    
    Returns (R_rot, R_orig_unrotated)
    """
    # Rotate weight(s). Handles (E,R,IN) or (R,IN) etc. - rotation over last dim.
    # Preserve leading dims: signs shape (IN,)
    W_rot = rotate_weights(weight, signs)
    # Note: col_weights are NOT rotated for incumbent control; we keep original
    # values as per-column weights in rotated space (they index rotated columns
    # by same position).  This is documented and tested under identity rotation.
    R_rot = quantize_fn(W_rot, col_weights)
    R_orig = unrotate_weights(R_rot, signs)
    return R_rot, R_orig


def identity_quantizer(weight_rot: torch.Tensor, col_weights: torch.Tensor) -> torch.Tensor:
    """Pass-through quantizer for testing identity rotation equivalence."""
    return weight_rot.clone()
