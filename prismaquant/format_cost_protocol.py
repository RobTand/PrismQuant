"""The costing seam: turn a Sensitivity Card + a format into an allocator candidate.

WHERE THIS SITS
---------------
``allocator_solver.py`` already defines the entire contract the optimizer needs::

    @dataclass
    class Candidate:
        fmt: str
        bits_per_param: float
        memory_bytes: int
        predicted_dloss: float

and the scalar model that fills the last field::

    def predicted_dloss(h_trace, weight_mse, gain=1.0) -> float:
        return 0.5 * float(h_trace) * float(weight_mse) * float(gain)

This module does not replace that seam -- it *feeds* it. A format plugin turns
``(SensitivityUnit, W, FormatDescriptor)`` into a :class:`CostComponents`, and
:meth:`CostComponents.to_predicted_dloss` collapses that to the one float the
knapsack DP consumes. The optimizer is untouched, so an arbitrary format menu
becomes an arbitrary list of plugins rather than a change to the solver.

THE THREE COST MODELS, IN INCREASING FIDELITY
---------------------------------------------
1. ``SCALAR`` -- ``0.5 * h_trace * weight_mse``. Exactly today's behaviour, and
   the fallback when a card carries no vectors. Reproducing this byte-identically
   from a card is the primary acceptance test for this module: same formula,
   same inputs, so any drift is a bug and not a modelling choice.

2. ``MARGINAL`` -- weight error weighted by the per-channel Fisher marginals
   instead of by a single scalar. The scalar model is the rank-0 collapse of the
   same object; this is the rank-1 reconstruction. It is strictly more
   descriptive per unit and costs ``out + in`` floats.

3. ``AQUA`` -- adds an activation-quantization term so that W4A4 and W4A8 stop
   being the same candidate. See below.

AQUA-AURA: WHY THE A-SIDE NEEDS ITS OWN SENSITIVITY
---------------------------------------------------
``h_trace`` is a *weight-space* curvature. An activation-quantization error is
an *input-side* perturbation ``x -> x + dx``, which reaches the loss as
``dy = W dx``. Multiplying an input-side error by a weight-space sensitivity is
a currency error of exactly the kind ``activation_fair_pricing.py`` is the
autopsy of, so this module refuses to do it.

Under a diagonal model the output perturbation is::

    E||W dx||^2 = sum_j  var(dx_j) * ||W[:, j]||^2

and it becomes a loss delta through the OUTPUT-space Fisher ``g_sq_sum[o]``::

    dLoss_a ~= 0.5 * sum_o  g_sq[o] * (W[o, :]^2 . var_dx)

which is what :func:`activation_dloss` computes. Note this uses ``g_sq_sum``,
never ``h_trace``.

STATUS: the A-side term is a **screening surrogate**. It is research-tier until
a served W4A4-vs-W4A8 A/B exists. ``activation_fair_pricing.py`` is deliberately
left untouched: superseding it is a promotion decision on evidence, not a
drive-by refactor.

WHAT THIS MODULE REFUSES TO DO
------------------------------
- Sum costs measured in different currencies (raises).
- Apply a passthrough format to a unit whose source dtype does not already match
  (BF16/FP8_SOURCE are passthrough-only; synthesising them wastes 8 bpp).
- Invent a speed/quality scalarization constant. Speed and quality are returned
  as separate axes; the choice between them is a frontier selection, not a
  weighted sum.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import math
from typing import Protocol

import numpy as np

from .sensitivity_card import Currency, RenderBasis, SensitivityUnit


class CostModel(enum.Enum):
    """Fidelity tier used to price a unit. Recorded on every cost."""

    SCALAR = "scalar"
    MARGINAL = "marginal"
    AQUA = "aqua"


@dataclasses.dataclass(frozen=True)
class FormatDescriptor:
    """Everything the costing seam needs to know about a format.

    This is intentionally a *description*, not an implementation: a downstream
    author naming a platform supplies these fields for their own formats without
    touching PrismaQuant. ``weight_bits`` and friends describe storage;
    ``act_bits`` describes what the kernel does to activations at serve time.
    """

    name: str
    #: Effective stored bits per weight parameter, INCLUDING scale/codebook
    #: overhead amortized over the group. This is what the byte budget spends.
    weight_bits: float
    #: WIDTH of the serve-time activation grid, when there is one. This is the
    #: quantity the error model needs; it is NOT the predicate for "does this
    #: format quantize activations" -- see :attr:`quantizes_activations`.
    #: Re-deriving that predicate from a width is the bug
    #: `test_activation_quant_predicate_has_one_definition` exists to prevent,
    #: because consumers that did so disagreed with the allocator's gate.
    act_bits: int | None = None
    #: THE predicate. Explicit data rather than an inference, sourced from
    #: ``FormatSpec.act_quant_changes_input`` by `format_cost_registry`. This
    #: field is the entire difference between W4A4 and W4A16.
    quantizes_activations: bool = False
    #: WEIGHT quantization group size along the input dimension, if any.
    group_size: int | None = None
    #: ACTIVATION quantization group size, i.e. how many input channels share
    #: one serve-time activation scale. Distinct from :attr:`group_size` on
    #: purpose: they coincide for NVFP4 (16/16) and MX (32/32), but a downstream
    #: format may block its weights and its activations differently, and the
    #: A-side error model needs the A-side number. ``None``/0 means the
    #: activation grid spans the whole channel (per-tensor or per-channel),
    #: which is what :func:`uniform_act_quant_variance` assumes.
    act_group_size: int | None = None
    #: True when the format is a verbatim copy of an already-matching source
    #: tensor (BF16, FP8_SOURCE). Legal only when the source dtype matches.
    passthrough: bool = False
    #: Source dtype this passthrough format requires, e.g. "bfloat16".
    requires_source_dtype: str | None = None
    #: Optional relative serve-time throughput hint, higher is faster. Used only
    #: to report the speed axis of a frontier; never folded into the loss.
    speed_index: float | None = None

    def is_legal_for(self, unit: SensitivityUnit) -> bool:
        """Passthrough integrity: never synthesize a passthrough format."""
        if not self.passthrough:
            return True
        if self.requires_source_dtype is None:
            return False
        return unit.topology.source_dtype == self.requires_source_dtype


@dataclasses.dataclass(frozen=True)
class CostComponents:
    """A priced (unit, format) pair, with the currency of every part declared."""

    unit_name: str
    format_name: str
    model: CostModel
    render_basis: RenderBasis

    #: Weight-side error, in WEIGHT_MSE currency.
    weight_mse: float
    #: Weight-side predicted loss delta, in DELTA_LOSS currency.
    weight_dloss: float
    #: Activation-side predicted loss delta, in DELTA_LOSS currency.
    #: ``None`` when the format does not quantize activations, or when the card
    #: lacks the vectors needed to price it -- which is NOT the same as zero and
    #: is kept distinct so a missing measurement never reads as "free".
    act_dloss: float | None = None

    bits_per_param: float = 0.0
    memory_bytes: int = 0
    #: Parameter count of the unit. Carried so a consumer can aggregate bpp and
    #: params-weighted speed without back-deriving it from memory_bytes, which
    #: is undefined for codebook formats reporting ``bits_per_param == 0``.
    n_params: int = 0
    speed_index: float | None = None
    #: Whether the priced format touches activations, copied from
    #: ``FormatDescriptor.quantizes_activations``. Carried rather than inferred
    #: for the same reason it is explicit on the descriptor: it is the ONLY way
    #: to tell "this format leaves activations alone, so act_dloss=None is
    #: correct" from "this format quantizes activations but we failed to price
    #: them, so act_dloss=None is a hole". Consumers that re-derived it from the
    #: cost model got W4A16 wrong.
    quantizes_activations: bool = False

    def to_predicted_dloss(self) -> float:
        """Collapse to the single float the knapsack DP sums.

        Only DELTA_LOSS is additive across units, which is why this is the only
        currency that may leave this module toward the solver.
        """
        total = self.weight_dloss
        if self.act_dloss is not None:
            total += self.act_dloss
        return float(total)

    def assert_currency(self, expected: Currency) -> None:
        if expected is not Currency.DELTA_LOSS:
            raise ValueError(
                f"{self.unit_name}/{self.format_name}: costs leave this module "
                f"in {Currency.DELTA_LOSS.value}; refusing to serve "
                f"{expected.value}. Mixing bases is the failure mode "
                "activation_fair_pricing.py documents.")


# --------------------------------------------------------------------- pricing


def weight_dloss_scalar(unit: SensitivityUnit, weight_mse: float,
                        gain: float = 1.0) -> float:
    """Today's model, unchanged: ``0.5 * h_trace * weight_mse * gain``.

    Kept as a named function so the byte-identical reproduction test has
    something to call, and so any divergence from `allocator_solver` is a
    one-line diff rather than a hunt.
    """
    return 0.5 * float(unit.h_trace) * float(weight_mse) * float(gain)


def weight_dloss_marginal(unit: SensitivityUnit, dw_sq: np.ndarray,
                          gain: float = 1.0) -> float:
    """Fisher-weighted weight error using the per-channel marginals.

    ``dw_sq`` is the elementwise squared weight error [out, in] the format would
    incur. Under the rank-1 reconstruction ``H ~= outer(row, col) / h_trace_raw``
    the loss delta is::

        0.5 * sum_{o,i} H[o,i] * dw_sq[o,i]
          ~= 0.5 * (row @ dw_sq @ col) / h_trace_raw

    which never forms ``H``. Falls back to the scalar model when the card has no
    vectors, so a card without them is degraded, not broken.
    """
    if not unit.has_vectors or unit.h_trace_raw <= 0.0:
        return weight_dloss_scalar(unit, float(np.mean(dw_sq)), gain)

    row = np.asarray(unit.fisher_row, dtype=np.float64)
    col = np.asarray(unit.fisher_col, dtype=np.float64)
    dw = np.asarray(dw_sq, dtype=np.float64)
    if dw.shape != (unit.out_features, unit.in_features):
        raise ValueError(
            f"{unit.topology.name}: dw_sq has shape {dw.shape}, expected "
            f"({unit.out_features}, {unit.in_features})")

    quad = float(row @ dw @ col) / unit.h_trace_raw
    # h_trace_raw is a token SUM; h_trace is the token MEAN. row/col are sums
    # too, so the quadratic form above is in "sum" units and needs the same
    # normalization the scalar path applies.
    return 0.5 * (quad / max(1, unit.n_tokens)) * float(gain)


def _cuda_reduce_device():
    """The CUDA device to reduce on, or None.

    The A-side's only heavy arithmetic is ``sum_o g[o] * sum_j W[o,j]^2 var[j]``
    -- a square and a weighted row-sum over EVERY parameter, once per candidate
    format. On a 35B-A3B that is ~70 G multiply-adds per format. Running it in
    numpy float64 on the host is a hot path on the CPU (principle 7), and it
    pays twice: the float64 promotion of a [256, 1024, 2048] expert tensor
    allocates 4 GiB before the multiply even starts.
    """
    try:
        import torch
    except Exception:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    return torch, torch.device("cuda")


def _weighted_row_sum(w, var, g) -> float:
    """``sum_o g[o] * sum_j w[o, j]^2 * var[j]`` for one row block.

    GPU when there is one, host numpy otherwise, with the SAME accumulation
    semantics either way: the elementwise square and product are float32 on
    device but the reduction accumulates in float64
    (``sum(dtype=torch.float64)``). Every term here is non-negative -- squares
    times variances -- so there is no cancellation and the float32 products
    carry ~1e-7 relative error into a float64 sum. `tests/
    test_packed_expert_aqua_marginals.py` pins the two paths against each other.

    TF32 is not a risk on this path: no ``matmul`` is used. The row-sum is an
    elementwise multiply plus a reduction, which is memory-bound at these
    shapes anyway, so the safe formulation is also the fast one.
    """
    torch, device = _cuda_reduce_device()
    if torch is None:
        w_sq = np.asarray(w, dtype=np.float64) ** 2
        return float(np.asarray(g, dtype=np.float64) @ (w_sq @ np.asarray(var, dtype=np.float64)))
    w_t = w if torch.is_tensor(w) else torch.from_numpy(np.ascontiguousarray(w))
    w_t = w_t.to(device=device, dtype=torch.float32, non_blocking=True)
    var_t = torch.as_tensor(np.asarray(var, dtype=np.float32), device=device)
    g_t = torch.as_tensor(np.asarray(g, dtype=np.float64), device=device)
    rows = (w_t * w_t * var_t).sum(dim=1, dtype=torch.float64)
    return float((g_t * rows).sum())


def _row_chunk(in_features: int, itemsize: int = 4) -> int:
    """Rows per block, bounding the reduction temporary to 256 MiB."""
    return max(1, int((256 << 20) // max(1, in_features * itemsize)))


def activation_dloss(unit: SensitivityUnit, weight: np.ndarray,
                     act_var: np.ndarray, gain: float = 1.0) -> float | None:
    """AQUA-AURA: loss delta from quantizing the layer's INPUT activations.

    ``act_var[j]`` is the per-input-channel variance of the activation
    quantization error. The perturbation reaches the loss as ``dy = W dx``, so
    under a diagonal model::

        dLoss ~= 0.5 * sum_o g_sq[o] * sum_j W[o,j]^2 * act_var[j]

    Returns ``None`` -- never 0.0 -- when the card lacks ``g_sq_sum``, so an
    unmeasured A-side is distinguishable from a free one.

    NOTE the sensitivity used here is ``g_sq_sum`` (output space), NOT
    ``h_trace`` (weight space). That distinction is the whole point.
    """
    if unit.has_expert_activation_stats:
        return _activation_dloss_packed(unit, weight, act_var, gain)
    if unit.g_sq_sum is None:
        return None
    g_sq = np.asarray(unit.g_sq_sum, dtype=np.float64)
    # NOT np.asarray: `weight` may be a device tensor the caller uploaded once
    # and reuses across formats, and np.asarray would drag it back to the host.
    w = weight
    var = np.asarray(act_var, dtype=np.float64)
    if tuple(w.shape) != (unit.out_features, unit.in_features):
        raise ValueError(f"{unit.topology.name}: weight shape mismatch")
    if var.shape != (unit.in_features,):
        raise ValueError(f"{unit.topology.name}: act_var shape mismatch")

    # Chunked over output rows: squaring the whole weight at once costs a full
    # extra copy -- 5.1 GiB for a 248320 x 5120 lm_head -- which is enough on
    # its own to exhaust a shared 121 GiB pool once a few formats are priced
    # back to back. The result is a sum over rows, so a row-blocked
    # accumulation is arithmetically the same quantity at a bounded peak.
    rows_per_chunk = _row_chunk(unit.in_features)
    total = 0.0
    for lo in range(0, unit.out_features, rows_per_chunk):
        hi = min(lo + rows_per_chunk, unit.out_features)
        total += _weighted_row_sum(w[lo:hi], var, g_sq[lo:hi])
    return 0.5 * (total / max(1, unit.n_tokens)) * float(gain)


def _activation_dloss_packed(unit: SensitivityUnit, weight: np.ndarray,
                             act_var: np.ndarray,
                             gain: float = 1.0) -> float | None:
    """``activation_dloss`` for a packed [E, M, N] routed-expert unit.

    The same quantity, summed over experts BEFORE the global normalization::

        dLoss ~= 0.5 / T_global
                 * sum_e sum_o g_sq[e, o] * sum_j W[e, o, j]^2 * var[e, j]

    Why this cannot be collapsed to the dense form: routing makes ``g_sq`` and
    ``var`` functions of e, and ``W`` is a different matrix per e, so the sum of
    the products is not the product of the sums. Two experts with identical
    aggregate statistics can differ by orders of magnitude in A-side once you
    account for WHICH tokens each one saw.

    The normalization asymmetry is deliberate and is the easiest thing to get
    wrong here. ``g_sq[e]`` is a RAW sum over expert e's routed tokens and is
    divided by the GLOBAL count, so a rarely-routed expert contributes little --
    correct, that is its share of the mean-delta-loss objective. ``var[e]`` is
    already a per-token mean (the caller fits it against ``expert_tokens[e]``),
    so it carries no frequency information at all. Fitting the variance
    globally too would discount a rare expert twice.
    """
    w = weight
    g_all = np.asarray(unit.expert_g_sq_sum, dtype=np.float64)
    var = np.asarray(act_var, dtype=np.float64)
    n_e = int(g_all.shape[0])
    if w.ndim != 3 or tuple(w.shape) != (
            n_e, unit.out_features, unit.in_features):
        raise ValueError(
            f"{unit.topology.name}: packed weight shape {w.shape}, expected "
            f"{(n_e, unit.out_features, unit.in_features)}")
    if var.shape == (unit.in_features,):
        # One population variance shared by every expert. Legal (the block
        # scale is a function of the token, not of the routing decision) and
        # explicitly broadcast rather than silently indexed.
        var = np.broadcast_to(var, (n_e, unit.in_features))
    elif var.shape != (n_e, unit.in_features):
        raise ValueError(
            f"{unit.topology.name}: packed act_var shape {var.shape}, expected "
            f"{(n_e, unit.in_features)} or {(unit.in_features,)}")

    # Chunked over output rows within each expert, same bound as the dense
    # path: a [256, 2048, 512] gate_up promoted to float64 whole is 2 GiB
    # before the temporary, and several formats are priced back to back.
    rows_per_chunk = _row_chunk(unit.in_features)
    total = 0.0
    for e in range(n_e):
        g_e = g_all[e]
        v_e = var[e]
        w_e = w[e]
        for lo in range(0, unit.out_features, rows_per_chunk):
            hi = min(lo + rows_per_chunk, unit.out_features)
            total += _weighted_row_sum(w_e[lo:hi], v_e, g_e[lo:hi])
    return 0.5 * (total / max(1, unit.n_tokens)) * float(gain)


def expert_act_sigma(unit: SensitivityUnit) -> np.ndarray | None:
    """Per-expert per-channel activation sigma, [E, N].

    ``expert_act_sq_sum[e]`` divided by expert e's OWN routed token count.
    An expert that saw zero calibration tokens yields a zero row: it has no
    measured activation distribution, and inventing one would put a fabricated
    A-side on the least-evidenced part of the model.
    """
    if unit.expert_act_sq_sum is None or unit.expert_tokens is None:
        return None
    sq = np.asarray(unit.expert_act_sq_sum, dtype=np.float64)
    tok = np.asarray(unit.expert_tokens, dtype=np.float64)
    denom = np.maximum(tok, 1.0)[:, None]
    return np.sqrt(sq / denom)


def price_activation_only(unit: SensitivityUnit, weight: np.ndarray,
                          plugin: FormatCostPlugin, *,
                          gain: float = 1.0,
                          act_var: np.ndarray | None = None,
                          rendered_weight=None) -> float | None:
    """The A-side price of one (unit, format).

    Split out of :func:`price` so there is exactly one implementation of "how
    much does quantizing this layer's activations cost", and so the A-side can
    be computed on its own.

    TWO WEIGHT BASES, AND THE CHOICE IS THE CALLER'S
    ------------------------------------------------
    By default the term is evaluated on the SOURCE weight ``W``: no render is
    needed, so an A-side priced next to an RTN render and one priced next to
    the production render are the same number, and a consumer holding a
    production weight cost can add this term to it without re-rendering
    anything. That is what makes the A-side shippable as its own stage.

    ``rendered_weight`` opts into the render-conditioned form. Pass the
    format's own rendered tensor ``W_hat_f`` and the term becomes
    ``sum_o g_sq[o] sum_j W_hat_f[o,j]^2 nu_j`` instead. The local decomposition
    is::

        W_hat x_hat - W x  =  dW x  +  W_hat dx

    so evaluating the activation term on ``W_hat_f`` rather than on ``W``
    absorbs the ``dW dx`` contribution into it, which the source-weight form
    drops. WHAT STAYS DROPPED, either way: the cross-correlation between
    ``dW x`` and ``W_hat dx`` (the W-side and A-side are summed as if
    independent), every downstream NONLINEAR interaction, and, on a routed MoE,
    the routing interaction -- a perturbed input can change which expert a token
    goes to, and no second-moment term sees a route flip. This is a screening
    surrogate on both bases; the served A/B is the result.

    The A-side also matters more than its size on an RTN basis suggests:
    GPTQ/JSO shrink the W-side substantially and do nothing whatever to the
    A-side, so the better the render, the more of a W4A4 format's true cost
    lives here.

    Returns ``None`` -- never 0.0 -- when the format leaves activations alone
    (nothing to price) or when the card cannot price them (a hole). Callers that
    need to tell those apart read ``descriptor.quantizes_activations``.
    """
    desc = plugin.descriptor
    if not desc.quantizes_activations:
        return None
    # Variance sources, most faithful first:
    #
    #   1. ``act_var`` supplied by the caller -- measured through this format's
    #      own quantizer on the layer's REAL cached input rows. Nothing is
    #      assumed about the activation distribution at all.
    #   2. the plugin's ``activation_error_variance`` -- the real quantizer, but
    #      over independent per-channel Gaussians fitted to the card's
    #      ``act_sq_sum``/``act_absmax``. Faithful marginals, no joint: it cannot
    #      see how outliers co-occur ACROSS the 16 consecutive channels that
    #      share one NVFP4 block scale.
    #   3. the analytic uniform-grid model -- assumes the grid shape too.
    #
    # Measured on Qwen3.8-27B across 32 stratified units, (2) is close to
    # unbiased against (1) -- median real/synthetic 1.011 for NVFP4 and 1.003
    # for FP8_E4M3 -- but the per-unit spread is real (p10 0.83 / p90 1.15 on
    # NVFP4, and 0.45 on L0 down_proj), which is why (1) exists as an option
    # rather than a claim that (2) is good enough everywhere.
    var = act_var
    if var is None and unit.has_expert_activation_stats:
        # A packed unit needs a variance PER EXPERT; the dense measure would
        # read `act_sq_sum`, which a packed entry does not have, and return
        # None -- reading as a hole rather than as the wrong shape.
        measure = getattr(plugin, "expert_activation_error_variance", None)
        if callable(measure):
            var = measure(unit)
    if var is None:
        measure = getattr(plugin, "activation_error_variance", None)
        if callable(measure):
            var = measure(unit)
    if var is None:
        var = analytic_act_quant_variance(unit, desc)
    if var is None:
        return None
    # `weight` still selects the variance model above (it never reads W), and
    # only the reduction switches basis. Passing the rendered tensor to the
    # variance fit would be meaningless -- nu is a property of the INPUT
    # distribution and the format's activation grid, not of the weights.
    return activation_dloss(
        unit, weight if rendered_weight is None else rendered_weight,
        var, gain)


def uniform_act_quant_variance(unit: SensitivityUnit, act_bits: int,
                               ) -> np.ndarray | None:
    """Per-channel activation-quantization error variance for a uniform grid.

    Uses ``act_absmax`` when the card carries it, because activation
    quantization error is driven by the dynamic range a channel actually spans;
    otherwise falls back to a Gaussian-equivalent range from ``act_sq_sum``.

    The 1/12 factor is the variance of a uniform distribution over one step --
    a property of the quantizer, not a tuned constant.
    """
    n_levels = float(2 ** act_bits)
    if unit.act_absmax is not None:
        rng = 2.0 * np.asarray(unit.act_absmax, dtype=np.float64)
    elif unit.act_sq_sum is not None:
        sigma = np.sqrt(np.asarray(unit.act_sq_sum, dtype=np.float64)
                        / max(1, unit.n_tokens))
        # A symmetric quantizer must span roughly +/-4 sigma to avoid clipping
        # dominating; this is the standard Gaussian-range surrogate used when a
        # true absmax was not captured, and it is why act_absmax is preferred.
        rng = 8.0 * sigma
    else:
        return None
    step = rng / n_levels
    return (step ** 2) / 12.0


#: Quadrature support for the folded-normal maximum. ``_QUAD_MAX`` is expressed
#: in units of the block's LARGEST channel sigma, so 20 is ~5x past where the
#: survival function is numerically zero regardless of absolute activation
#: scale. ``_ERF_POINTS`` tabulates ``erf(u/sqrt2)`` densely enough that linear
#: interpolation is exact to ~1e-7.
_QUAD_MAX = 20.0
_ERF_POINTS = 20001
_GL_NODES = 64


@functools.lru_cache(maxsize=1)
def _folded_normal_table() -> tuple[np.ndarray, np.ndarray]:
    """Abscissae and ``erf(u/sqrt2)``, built once on first use.

    Interpolating this table at ``u/s`` returns ``erf(u/(s*sqrt2))`` -- the fold
    is already baked in, so callers must NOT divide by ``sqrt2`` again. Doing so
    inflates ``E[M^2]`` by exactly 2x, which is silent because the result stays
    dimensionally plausible.
    """
    u = np.linspace(0.0, _QUAD_MAX, _ERF_POINTS)
    erf_u = np.array([math.erf(v / math.sqrt(2.0)) for v in u])
    return u, erf_u


@functools.lru_cache(maxsize=1)
def _gauss_legendre() -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights mapped onto ``[0, _QUAD_MAX]``."""
    x, w = np.polynomial.legendre.leggauss(_GL_NODES)
    return 0.5 * _QUAD_MAX * (x + 1.0), 0.5 * _QUAD_MAX * w


def _expected_blockmax_sq(sigma: np.ndarray) -> np.ndarray:
    """``E[max_j (sigma_j z_j)^2]`` per block, for independent normal channels.

    ``sigma`` is ``[n_blocks, G]``; a ragged final block is padded with
    ``sigma = 0``, which is exactly right rather than merely convenient -- a
    zero-scale channel contributes ``erf(inf) = 1`` to the product and so does
    not affect the maximum.

    Uses the tail-integral identity for a non-negative variable::

        E[M^2] = int_0^inf 2u (1 - P(M <= u)) du,
        P(M <= u) = prod_j erf(u / (sigma_j * sqrt2))

    which is EXACT for independent Gaussian channels of differing scale -- no
    order-statistic asymptotic, and in particular no assumption that the
    channels within a block share a scale. That last point is what makes this
    worth the quadrature: collapsing a block to its RMS sigma (the obvious
    cheap move) under-prices real activation blocks badly, by 1.9x at a
    moderate lognormal spread, 2.5x at a wide one, and 3.5x when one channel
    dominates its block. This form is within 0.1% of simulation in all four
    regimes.
    """
    nodes, weights = _gauss_legendre()
    table_u, table_erf = _folded_normal_table()

    # Work in units of each block's largest sigma so the fixed integration
    # range always covers the tail. In absolute units a block whose sigma
    # approaches _QUAD_MAX/4 truncates and silently under-integrates.
    s_max = sigma.max(axis=1, keepdims=True)
    safe_max = np.where(s_max > 0.0, s_max, 1.0)
    ratio = np.clip(sigma / safe_max, 1e-12, None)

    arg = nodes[None, None, :] / ratio[:, :, None]
    erf_v = np.interp(arg, table_u, table_erf)
    # Product over channels in log space: a block of 32 near-1.0 factors
    # multiplies to something that underflows far less gracefully in the direct
    # form once sigma ratios are extreme.
    survival = 1.0 - np.exp(np.log(np.clip(erf_v, 1e-300, 1.0)).sum(axis=1))
    normalized = (2.0 * nodes[None, :] * survival) @ weights
    return normalized * s_max[:, 0] ** 2


def block_scaled_act_quant_variance(unit: SensitivityUnit, act_bits: int,
                                    act_group_size: int,
                                    ) -> np.ndarray | None:
    """AQUA-1: per-channel activation error variance for a BLOCK-SCALED grid.

    :func:`uniform_act_quant_variance` sets the step from ``act_absmax[j]`` --
    the largest value channel ``j`` reached over the WHOLE calibration. That is
    right for a per-tensor/per-channel grid and wrong for every format in the
    shipped menu, because NVFP4 (G=16) and MX (G=32) rescale *per block, per
    token*: the step follows the local block maximum, which is far below the
    global one. Pricing a block-scaled quantizer with a global step therefore
    OVER-states its error, and it does so asymmetrically -- it penalises exactly
    the W4A4 formats, herding the allocator toward W4A8 for a modelling reason
    rather than a measured one.

    The step is set by the expected within-block maximum, which
    :func:`_expected_blockmax_sq` computes exactly from the per-channel scales
    ``sigma_j^2 = act_sq_sum[j] / n_tokens``. The familiar ``sqrt(2 ln 2G)``
    asymptotic is deliberately NOT used: at the block sizes that actually ship
    it is wrong by +52% (G=16) and +46% (G=32) against a 400k-sample
    simulation, and its standard Fisher-Tippett correction only turns that into
    -14%. An exact expression exists, so per the no-heuristics rule the exact
    expression is what runs.

    ASSUMPTIONS, all of which degrade gracefully and none of which are hidden:

    * **Gaussian and independent within a block.** Real activations are heavy
      tailed and correlated across channels. A heavy tail makes the true block
      max *larger* than this estimate, so the model is optimistic there -- the
      opposite direction to the uniform model's pessimism, and far smaller in
      magnitude. Per-channel heterogeneity is NOT an assumption: it is carried
      exactly.
    * **Dynamic, per-token block scales.** This prices what NVFP4/MX kernels do
      at serve time. A statically calibrated activation grid would be closer to
      :func:`uniform_act_quant_variance`, and a format declaring one should say
      so by leaving ``act_group_size`` unset.
    * **The block scale is treated as exact.** NVFP4 snaps it to FP8 and MX to a
      power of two; that quantization of the scale itself is second order and is
      not modelled here.

    Returns ``None`` when the card carries no ``act_sq_sum`` -- the block model
    needs per-channel *scales*, and ``act_absmax`` alone cannot supply them
    without re-introducing the global-max error this function exists to remove.
    """
    if unit.act_sq_sum is None:
        return None
    sigma_sq = (np.asarray(unit.act_sq_sum, dtype=np.float64)
                / max(1, unit.n_tokens))
    n_in = int(sigma_sq.shape[0])
    group = min(int(act_group_size), n_in)
    if group <= 0:
        return None

    # Pad to a whole number of blocks with sigma = 0. A zero-scale channel is
    # inert in the max, so the ragged tail is handled exactly rather than by a
    # separate code path.
    n_blocks = -(-n_in // group)                      # ceil
    padded = np.zeros(n_blocks * group, dtype=np.float64)
    padded[:n_in] = np.sqrt(sigma_sq)
    expected_blockmax_sq = _expected_blockmax_sq(padded.reshape(n_blocks, group))

    step = 2.0 * np.sqrt(expected_blockmax_sq) / float(2 ** act_bits)
    var_per_block = (step ** 2) / 12.0
    # Every channel in a block shares that block's scale, hence its step, hence
    # its error variance. This per-channel broadcast is what activation_dloss
    # consumes; trim the padding back off.
    return np.repeat(var_per_block, group)[:n_in]


def analytic_act_quant_variance(unit: SensitivityUnit,
                                desc: "FormatDescriptor",
                                ) -> np.ndarray | None:
    """Pick the analytic A-side error model that matches the format's grid.

    Block-scaled formats get :func:`block_scaled_act_quant_variance`; a format
    that declares no activation grouping keeps the per-channel/global model.
    Dispatching on declared metadata rather than on the format's name keeps an
    arbitrary downstream format correctly priced without editing this file.
    """
    if desc.act_bits is None:
        return None
    if desc.act_group_size:
        return block_scaled_act_quant_variance(
            unit, desc.act_bits, desc.act_group_size)
    return uniform_act_quant_variance(unit, desc.act_bits)


class FormatCostPlugin(Protocol):
    """What a format must implement to be priced by an arbitrary consumer.

    A downstream author adds a format by supplying one of these -- no change to
    the probe, the card, or the solver.
    """

    descriptor: FormatDescriptor

    def weight_error(self, unit: SensitivityUnit,
                     weight: np.ndarray) -> np.ndarray:
        """Elementwise squared weight error [out, in] this format would incur.

        Computed from the weight alone under the card's declared render basis
        (RTN for a shareable card). No Hessian, no calibration replay.
        """
        ...


def price(unit: SensitivityUnit, weight: np.ndarray,
          plugin: FormatCostPlugin, *, render_basis: RenderBasis,
          model: CostModel = CostModel.MARGINAL,
          gain: float = 1.0) -> CostComponents | None:
    """Price one (unit, format) pair. Returns None when the format is illegal.

    Legality is passthrough integrity only -- a format is never rejected here
    for looking risky. Banning formats in the coster is the "post-allocator
    rewrite" antipattern; the platform bounds error, it does not restrict what
    the allocator may consider.
    """
    desc = plugin.descriptor
    if not desc.is_legal_for(unit):
        return None

    # Prefer a plugin that can REDUCE the weight error without handing back a
    # dense [out, in] array. Both quantities below are reductions, so the dense
    # array is pure overhead -- and on a large-vocab lm_head it is gigabytes of
    # it, in device memory that host RSS does not even show. A plugin without
    # this method, or one returning None (no marginals to contract against),
    # falls through to the dense path unchanged.
    reduced = None
    if model is not CostModel.SCALAR:
        reduce_fn = getattr(plugin, "weight_cost_reduced", None)
        if callable(reduce_fn):
            reduced = reduce_fn(unit, weight)

    if reduced is not None:
        weight_mse, quad = reduced
        if unit.has_vectors and unit.h_trace_raw > 0.0:
            w_dloss = 0.5 * (quad / unit.h_trace_raw
                             / max(1, unit.n_tokens)) * float(gain)
        else:
            w_dloss = weight_dloss_scalar(unit, weight_mse, gain)
        dw_sq = None
    else:
        dw_sq = plugin.weight_error(unit, weight)
        weight_mse = float(np.mean(dw_sq))
        if model is CostModel.SCALAR:
            w_dloss = weight_dloss_scalar(unit, weight_mse, gain)
        else:
            w_dloss = weight_dloss_marginal(unit, dw_sq, gain)

    a_dloss: float | None = None
    if model is CostModel.AQUA:
        a_dloss = price_activation_only(unit, weight, plugin, gain=gain)

    return CostComponents(
        unit_name=unit.topology.name,
        format_name=desc.name,
        model=model,
        render_basis=render_basis,
        weight_mse=weight_mse,
        weight_dloss=w_dloss,
        act_dloss=a_dloss,
        bits_per_param=desc.weight_bits,
        memory_bytes=int(round(unit.n_params * desc.weight_bits / 8.0)),
        n_params=int(unit.n_params),
        speed_index=desc.speed_index,
        quantizes_activations=bool(desc.quantizes_activations),
    )
