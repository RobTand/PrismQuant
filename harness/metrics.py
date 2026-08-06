"""Weighted / unweighted MSE and gap metrics for rotation study."""
from __future__ import annotations

import torch


def weighted_mse(weight: torch.Tensor, recon: torch.Tensor, col_weights: torch.Tensor) -> torch.Tensor:
    """Per-expert column-weighted MSE, matching production metric.
    
    weight, recon: (E, R, IN) or (E, OUT, IN) or (OUT, IN) stacked;
    col_weights: (E, 1, IN) or (E, IN) or (1, IN) broadcastable to weight.
    Returns per-expert scalar tensor (E,) or scalar if single.
    
    Mirrors prismaquant layer_streaming / tools/dsv4_ldlq_cost_campaign.py:238-248
    per_slice_weighted_mse:
      err2 = (w - recon).square
      cw = broadcast(col_weights, err2.shape)
      return (err2 * cw).sum / cw.sum
    """
    err2 = (weight.to(torch.float32) - recon.to(torch.float32)).pow(2)
    cw = col_weights.to(torch.float32)
    # Broadcast cw to err2 shape: handle (E,1,IN) -> (E,R,IN) etc.
    try:
        cw = torch.broadcast_to(cw, err2.shape)
    except RuntimeError:
        # Try expanding trailing dim
        cw = cw.expand_as(err2)
    denom = cw.sum(dim=tuple(range(1, err2.dim()))) if err2.dim() > 1 else cw.sum()
    # For 3D case (E,R,IN): sum over R,IN
    if err2.dim() == 3:
        # (E,R,IN) -> per-E
        num = (err2 * cw).sum(dim=(1, 2))
        den = cw.sum(dim=(1, 2)).clamp_min(1e-30)
        return num / den
    elif err2.dim() == 2:
        # (R,IN) single expert
        return ((err2 * cw).sum() / cw.sum().clamp_min(1e-30)).unsqueeze(0)
    else:
        # fallback
        return ((err2 * cw).sum() / cw.sum().clamp_min(1e-30)).unsqueeze(0)


def unweighted_mse(weight: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Per-expert unweighted MSE (mean over all elements)."""
    err2 = (weight.to(torch.float32) - recon.to(torch.float32)).pow(2)
    if err2.dim() == 3:
        return err2.mean(dim=(1, 2))
    elif err2.dim() == 2:
        return err2.mean().unsqueeze(0)
    else:
        return err2.mean().unsqueeze(0)


def gap_closed_percent(a_mse: float, b_mse: float, c_mse: float) -> float:
    """% of (A-B) gap closed by C: (A-C)/(A-B) * 100.
    
    If C==B => 100% (fully closes), C==A => 0%, negative if C worse than A.
    Returns NaN if A==B (no gap).
    """
    denom = float(a_mse) - float(b_mse)
    if abs(denom) < 1e-30:
        return float("nan")
    return (float(a_mse) - float(c_mse)) / denom * 100.0
