"""Hierarchical rate-distortion law for FP8 product codebook error.

Level 1 (global per-projection): log G(K) = a0 + a1*K + phi_{K%4}
  where phi_0=0, phi_1,phi_2,phi_3 are family offsets (4 residue families).
  Equivalent to error(K)=C * rho^K * s_{K%4} with rho=exp(a1).
  This is the identifiable form of error(K)=sum_i c_i rho^{bits_i(K)}
  with single rho (rho_i=rho for all i).  Free per-family rho is degenerate
  (ill-conditioned, 0.98 correlation) and does not improve K28-38 fit.

Level 2 (per layer x projection): log L(K) = log G(K) + a + b*K
  two DoF (scale a and tilt b) fitted from n anchors via linear LS in log space.
  a captures layer-wide scale, b captures slope difference (steepness).

Level 3 (per expert): residual handled by existing backstop; law nial MEDIANS.

References:
  prismaquant/cb_layout.py:bit_split, subtable_bit_widths
  DIAGNOSIS.md geometry table, OMEGA_FIX_EVAL.md (4-param NNLS failure)
"""
from __future__ import annotations
import numpy as np

try:
    from prismaquant.cb_layout import subtable_bit_widths  # noqa: F401
except Exception:
    subtable_bit_widths = None  # type: ignore

# Global Level-1 coefficients per projection fitted on L14 (21 rungs -> K28-38 medians)
# Obtained via LS on log median: log y = a0 + a1*K + phi_{r}
# See law_validate.py:fit_global()
_GLOBAL_COEFF = {
    # gate_proj: a0=-6.82492058 a1=-0.15897462 phi1=0.03593618 phi2=0.05849172 phi3=0.06637451
    "gate_proj": np.array([-6.82492058, -0.15897462, 0.03593618, 0.05849172, 0.06637451]),
    # up_proj: a0=-6.78727488 a1=-0.15963589 phi1=0.04178647 phi2=0.06700044 phi3=0.06039119
    "up_proj": np.array([-6.78727488, -0.15963589, 0.04178647, 0.06700044, 0.06039119]),
    # down_proj: a0=-6.47337165 a1=-0.16091762 phi1=0.04077672 phi2=0.06230697 phi3=0.05701946
    "down_proj": np.array([-6.47337165, -0.16091762, 0.04077672, 0.06230697, 0.05701946]),
}

# Uncertainties (bootstrap 1k resamples of per-expert medians, 95% CI)
# gate: a0 ±0.04, a1 ±0.0012, phi ±0.01; up/down similar.  Tight because 256 experts.
_GLOBAL_COEFF_STD = {
    "gate_proj": np.array([0.04, 0.0012, 0.01, 0.01, 0.01]),
    "up_proj": np.array([0.04, 0.0012, 0.01, 0.01, 0.01]),
    "down_proj": np.array([0.04, 0.0012, 0.01, 0.01, 0.01]),
}


def global_G(proj: str, K: int) -> float:
    """Level-1 global prediction G(K) for projection."""
    coeff = _GLOBAL_COEFF[proj]
    a0, a1, phi1, phi2, phi3 = coeff
    r = int(K) % 4
    phi = 0.0 if r == 0 else [phi1, phi2, phi3][r - 1]
    return float(np.exp(a0 + a1 * int(K) + phi))


def log_G(proj: str, K: int) -> float:
    coeff = _GLOBAL_COEFF[proj]
    a0, a1, phi1, phi2, phi3 = coeff
    r = int(K) % 4
    phi = 0.0 if r == 0 else [phi1, phi2, phi3][r - 1]
    return float(a0 + a1 * int(K) + phi)


def fit_global_from_medians(ks, ys, proj_hint: str = "gate_proj"):
    """Fit Level-1 global coeffs from median curve (ks 28..38, ys medians).

    Solves log y = a0 + a1*K + phi_{r}  (phi0=0) via LS.
    Returns coeff array [a0,a1,phi1,phi2,phi3].
    """
    ks = np.asarray(ks, dtype=float)
    ys = np.asarray(ys, dtype=float)
    logs = np.log(ys)
    n = len(ks)
    F = np.zeros((n, 5))
    F[:, 0] = 1
    F[:, 1] = ks
    for i, k in enumerate(ks):
        r = int(k) % 4
        if r == 1:
            F[i, 2] = 1
        elif r == 2:
            F[i, 3] = 1
        elif r == 3:
            F[i, 4] = 1
    coeff, *_ = np.linalg.lstsq(F, logs, rcond=None)
    return coeff


def fit_level2(anchors, anchor_medians, proj: str):
    """Fit per-layer Level-2 (a,b) from n anchors.

    anchors: iterable of K (int)
    anchor_medians: dict K->median or array aligned with anchors
    Returns (a,b) where log L(K)=log G(K)+a+b*K
    """
    if isinstance(anchor_medians, dict):
        ks = np.array(list(anchors), dtype=float)
        ys = np.array([float(anchor_medians[k]) for k in anchors], dtype=float)
    else:
        ks = np.array(list(anchors), dtype=float)
        ys = np.asarray(anchor_medians, dtype=float)
    # rhs = log y - log G
    rhs = np.log(ys) - np.array([log_G(proj, int(k)) for k in ks])
    A = np.vstack([np.ones_like(ks), ks]).T
    coeff, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return float(coeff[0]), float(coeff[1])


def predict(proj: str, K: int, a: float, b: float) -> float:
    """Level-2 prediction for rung K."""
    return float(global_G(proj, int(K)) * np.exp(a + b * int(K)))


def predict_curve(proj: str, ks, a: float, b: float):
    return np.array([predict(proj, int(k), a, b) for k in ks])


# Alternative original sum form for reference (fixed rho=0.5 degenerate)
def bits_vec(K: int):
    if subtable_bit_widths is None:
        # fallback ceil-first split
        k = int(K)
        base, extra = divmod(k, 4)
        return tuple(base + (1 if i < extra else 0) for i in range(4))
    return subtable_bit_widths(int(K), "product", 4)


def sum_model_fixed_rho(K: int, c, rho=0.5):
    b = bits_vec(K)
    return float(sum(ci * (rho ** bi) for ci, bi in zip(c, b)))
