"""Throughput-optimal draft selector for MTP / spec-decode modules — canon.

Normative spec: ``docs/design/mtp_rung_selection.md`` (Robert, 2026-07-20). A draft can
NEVER change outputs (rejection sampling reproduces the target distribution
exactly), so this selector optimises **throughput only** — there is no quality
gate on the draft. The reference integration is
``scripts/build_hy3_mtp_cb_inputs.py --rung-select auto``.

The original :func:`select_rung` model applies to sequential drafters whose
cost is ``t + k*d(b)``.  Block-parallel drafters such as DeepSeek DSpark issue
all ``k`` positions in one backbone call; they must use
:func:`select_measured_configuration`, which optimizes directly over measured
cycle time and per-position survival rather than pretending the block cost is
``k`` independent forwards.

Sequential objective (per spec-decode cycle, ``k`` speculative tokens; doc §1):

    T(b) = (1 + Σ_{i=1..k} Π_{j<=i} a_j(b)) / (t + k·d(b))

with per-position acceptance approximated as ``a(b)^i`` for position ``i`` (see
``_throughput``). Cost side is exact: ``d(b) = d0 + c·b``. Acceptance side is the
Fisher/Pinsker shape ``a(b) = a_inf − β·sqrt(E(b))`` with ``E(b) = Σ_i h_i·MSE_i``
per rung, and ``(a_inf, β)`` **fit from served acceptance measurements**.

This module is pure-Python (stdlib + optional scipy for the Lambert-W
cross-check); it never imports torch, so it stays importable in the CPU driver.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Optional

_LN2 = math.log(2.0)
# Doc §3.6: degenerate iff the cost side varies < 1% across the menu. This is
# the spec constant, not a tunable heuristic (exposed only for testability).
_DEGENERATE_FRACTION = 0.01


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RungPoint:
    """One draft-menu candidate: an aggregate draft encoding.

    ``bits`` is the params-weighted draft bpw ``b``; ``resident_bytes`` is the
    draft's serve-time footprint at this rung (feeds the memory gate);
    ``E`` = ``Σ_i h_i·MSE_i`` is the calibrated error proxy that drives ``a(b)``.
    """

    name: str
    bits: float
    resident_bytes: int
    E: float

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError(f"RungPoint {self.name}: bits must be > 0")
        if self.resident_bytes < 0:
            raise ValueError(f"RungPoint {self.name}: resident_bytes < 0")
        if self.E < 0 or not math.isfinite(self.E):
            raise ValueError(f"RungPoint {self.name}: E must be finite and >= 0")


@dataclass(frozen=True)
class ServeConstants:
    """Cost-side constants (doc §2). All times in ms.

    ``t_ms`` = target verify-step time; ``d0_ms`` = rung-independent drafter
    overhead (shared lm_head read + attention + KV + host/launch — host-dominated
    on an eager drafter); ``c_ms_per_bit`` = drafter time per bit/weight so that
    ``d(b) = d0_ms + c_ms_per_bit·b``.
    """

    t_ms: float
    d0_ms: float
    c_ms_per_bit: float

    def __post_init__(self) -> None:
        if self.t_ms <= 0:
            raise ValueError("ServeConstants: t_ms must be > 0")
        if self.d0_ms < 0 or self.c_ms_per_bit < 0:
            raise ValueError("ServeConstants: d0_ms and c_ms_per_bit must be >= 0")

    def d(self, bits: float) -> float:
        """One drafter forward at ``bits`` bits/weight (ms)."""
        return self.d0_ms + self.c_ms_per_bit * bits


@dataclass(frozen=True)
class AcceptancePoint:
    """A served acceptance measurement at one rung.

    Identify the rung by ``rung_name`` (matched against ``RungPoint.name`` and
    ``E_by_bits`` keys) OR by ``bits``; at least one is required. The fit maps a
    point to its ``E`` via ``key = bits if bits is not None else rung_name``.
    """

    measured_acceptance: float
    rung_name: Optional[str] = None
    bits: Optional[float] = None

    def __post_init__(self) -> None:
        if self.rung_name is None and self.bits is None:
            raise ValueError("AcceptancePoint: give rung_name or bits")
        if not (0.0 <= self.measured_acceptance <= 1.0):
            raise ValueError(
                f"AcceptancePoint: acceptance {self.measured_acceptance} "
                "outside [0, 1]")

    @property
    def key(self):
        """Resolution key into ``E_by_bits``: ``bits`` if set, else ``rung_name``."""
        return self.bits if self.bits is not None else self.rung_name


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
@dataclass
class SelectionResult:
    rung: RungPoint
    regime: str  # "degenerate" | "interior"
    per_rung_T: dict
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DraftMemoryLedger:
    """Explicit serve-memory constraint for a draft configuration.

    Every field is a resident or peak byte term in the same usable memory
    pool.  ``fixed_runtime_bytes`` is the target model/process footprint after
    removing the separately listed KV and profiling terms.  The candidate's
    own ``resident_bytes`` and ``peak_scratch_bytes`` are added by the selector.

    ``admission_mode`` is provenance, not an escape hatch.  A test may
    deliberately use ``"test-only-relaxed"`` with a zero safety margin, while
    a publishable selection must carry the production ledger.
    """

    usable_pool_bytes: int
    fixed_runtime_bytes: int
    target_kv_bytes: int = 0
    draft_kv_bytes: int = 0
    profiling_peak_bytes: int = 0
    safety_margin_bytes: int = 0
    admission_mode: str = "production"

    def __post_init__(self) -> None:
        if self.usable_pool_bytes <= 0:
            raise ValueError("DraftMemoryLedger: usable_pool_bytes must be > 0")
        for name in (
            "fixed_runtime_bytes", "target_kv_bytes", "draft_kv_bytes",
            "profiling_peak_bytes", "safety_margin_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"DraftMemoryLedger: {name} must be a non-negative int"
                )
        if not str(self.admission_mode).strip():
            raise ValueError("DraftMemoryLedger: admission_mode must be non-empty")

    @property
    def fixed_bytes(self) -> int:
        """All non-candidate terms, including the declared safety reserve."""

        return (
            self.fixed_runtime_bytes
            + self.target_kv_bytes
            + self.draft_kv_bytes
            + self.profiling_peak_bytes
            + self.safety_margin_bytes
        )

    @property
    def candidate_budget_bytes(self) -> int:
        """Bytes left for draft residency plus candidate-specific scratch."""

        return self.usable_pool_bytes - self.fixed_bytes


@dataclass(frozen=True)
class MeasuredDraftConfiguration:
    """One directly measured ``(quantization assignment, k)`` candidate.

    ``position_survival[i]`` is the cumulative probability that draft position
    ``i`` is accepted, measured as accepted-at-position counter delta divided
    by spec-decode cycle/draft counter delta.  It is cumulative (and therefore
    non-increasing), not the conditional acceptance at that position.

    ``cycle_ms`` is the complete served decode-cycle wall time at the fixed
    workload named by ``measurement_id``.  For a block-parallel drafter this is
    the architecture-faithful denominator; callers must not multiply it by
    ``k``.  ``load_ms`` is optional startup cost.  ``peak_scratch_bytes`` is
    candidate-specific peak memory above steady draft residency.
    """

    name: str
    k: int
    resident_bytes: int
    cycle_ms: float
    position_survival: tuple[float, ...]
    measurement_id: str
    load_ms: float = 0.0
    peak_scratch_bytes: int = 0
    bits: Optional[float] = None
    E: Optional[float] = None
    measurement_source: str = "unknown"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("MeasuredDraftConfiguration: name must be non-empty")
        if self.k < 1:
            raise ValueError(
                f"MeasuredDraftConfiguration {self.name}: k must be >= 1"
            )
        survival = tuple(float(value) for value in self.position_survival)
        object.__setattr__(self, "position_survival", survival)
        if len(survival) != self.k:
            raise ValueError(
                f"MeasuredDraftConfiguration {self.name}: position_survival "
                f"has {len(survival)} entries, expected k={self.k}"
            )
        for index, value in enumerate(survival):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"MeasuredDraftConfiguration {self.name}: survival[{index}] "
                    f"must be finite and in [0, 1], got {value}"
                )
            if index and value > survival[index - 1] + 1e-12:
                raise ValueError(
                    f"MeasuredDraftConfiguration {self.name}: cumulative "
                    "position survival must be non-increasing"
                )
        if self.resident_bytes < 0 or self.peak_scratch_bytes < 0:
            raise ValueError(
                f"MeasuredDraftConfiguration {self.name}: memory bytes < 0"
            )
        if self.cycle_ms <= 0 or not math.isfinite(self.cycle_ms):
            raise ValueError(
                f"MeasuredDraftConfiguration {self.name}: cycle_ms must be "
                "finite and > 0"
            )
        if self.load_ms < 0 or not math.isfinite(self.load_ms):
            raise ValueError(
                f"MeasuredDraftConfiguration {self.name}: load_ms must be "
                "finite and >= 0"
            )
        if self.bits is not None and (
            self.bits <= 0 or not math.isfinite(self.bits)
        ):
            raise ValueError(
                f"MeasuredDraftConfiguration {self.name}: bits must be finite "
                "and > 0 when present"
            )
        if self.E is not None and (self.E < 0 or not math.isfinite(self.E)):
            raise ValueError(
                f"MeasuredDraftConfiguration {self.name}: E must be finite "
                "and >= 0 when present"
            )
        if not str(self.measurement_id).strip():
            raise ValueError(
                f"MeasuredDraftConfiguration {self.name}: measurement_id must "
                "be non-empty"
            )
        if not str(self.measurement_source).strip():
            raise ValueError(
                f"MeasuredDraftConfiguration {self.name}: measurement_source "
                "must be non-empty"
            )

    @property
    def expected_tokens_per_cycle(self) -> float:
        """One verified target token plus expected accepted draft tokens."""

        return 1.0 + math.fsum(self.position_survival)

    @property
    def steady_state_tokens_per_second(self) -> float:
        return 1000.0 * self.expected_tokens_per_cycle / self.cycle_ms

    @property
    def candidate_peak_bytes(self) -> int:
        return self.resident_bytes + self.peak_scratch_bytes


@dataclass
class MeasuredConfigurationSelection:
    configuration: MeasuredDraftConfiguration
    per_configuration: dict
    pareto_frontier: tuple[str, ...]
    provenance: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Acceptance fit:  a(b) = a_inf − β·sqrt(E(b))   (linear in sqrt(E); doc §2)
# --------------------------------------------------------------------------- #
def fit_acceptance(points, E_by_bits: Mapping):
    """Fit ``(a_inf, β)`` from served acceptance measurements.

    Returns ``(a_inf, beta, fit_mode)`` where ``fit_mode`` is:

      * ``"least_squares"`` — ≥2 points with distinct ``sqrt(E)``: ordinary
        least-squares line ``a = a_inf − β·x`` over ``x = sqrt(E)``.
      * ``"single_point"`` — exactly 1 point, OR ≥2 points that all share one
        ``E`` (no fidelity spread → no slope). ``a_inf`` is taken from the
        highest-fidelity (lowest-E) measured rung; ``beta`` is ``None`` (there is
        no slope to estimate — the doc forbids assuming one). The selector must
        then fall back to the degenerate branch.
      * ``"no_data"`` — 0 points: ``a_inf = beta = None``.

    ``E_by_bits`` maps each point's ``key`` (its ``bits`` if set, else
    ``rung_name``) to ``E(b)``; a missing key is a hard error (never fabricate E).
    """
    pts = list(points)
    if not pts:
        return None, None, "no_data"

    xs, ys = [], []
    for p in pts:
        if p.key not in E_by_bits:
            raise KeyError(
                f"fit_acceptance: no E for acceptance point key {p.key!r}")
        E = float(E_by_bits[p.key])
        if E < 0 or not math.isfinite(E):
            raise ValueError(f"fit_acceptance: bad E={E} for key {p.key!r}")
        xs.append(math.sqrt(E))
        ys.append(float(p.measured_acceptance))

    n = len(pts)
    x_span = max(xs) - min(xs)
    # Single point, or a degenerate cluster with no fidelity spread → no slope.
    if n == 1 or x_span <= 1e-12:
        # a_inf := acceptance at the highest-fidelity (lowest-E ⇒ lowest-x) rung;
        # average ties so a repeated-rung calibration is not order-dependent.
        x_min = min(xs)
        best = [y for x, y in zip(xs, ys) if abs(x - x_min) <= 1e-12]
        return sum(best) / len(best), None, "single_point"

    # Ordinary least squares for the line a = intercept + slope·x (β = −slope).
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    return intercept, -slope, "least_squares"


# --------------------------------------------------------------------------- #
# Throughput and the continuous cross-check
# --------------------------------------------------------------------------- #
def _acceptance(a_inf, beta, E: float) -> float:
    """a(b) = a_inf − β·sqrt(E), clamped to the probability domain [0, 1].

    Clamping enforces that acceptance is a probability (a mathematical bound, not
    a heuristic band-aid on the optimiser); ``provenance['a_clamped']`` records
    whether any menu rung's fitted value fell outside [0, 1].
    """
    a = a_inf - (beta or 0.0) * math.sqrt(max(E, 0.0))
    return min(1.0, max(0.0, a))


def _throughput(a: float, d: float, t: float, k: int) -> float:
    """T = (1 + Σ_{i=1..k} a^i) / (t + k·d).

    Per-position cumulative acceptance ``Π_{j<=i} a_j`` is approximated by
    ``a^i`` (all positions share the fitted ``a(b)``; doc §1). k=1 reduces to
    ``(1 + a) / (t + d)``.
    """
    num, term = 1.0, 1.0
    for _ in range(k):
        term *= a
        num += term
    return num / (t + k * d)


def _continuous_bstar_fixed_point(a_inf, beta, const: ServeConstants,
                                  b0: float, iters: int = 3):
    """Lambert-W continuous optimum via fixed-point iteration (scipy-free).

    Solves the k=1 stationarity condition (doc §3.4)
        2^{-b}·β·[ln2·(t+d0+c·b) + c] = c·(1 + a_inf)
    rearranged to b = (1/ln2)·ln( β·[ln2·(t+d0+c·b)+c] / (c·(1+a_inf)) ). The
    RHS depends on b only through a log, so 2–3 iterations converge. Theory /
    sanity cross-check on the discrete argmax — never the shipped selection.
    Returns None when the curve has no interior optimum (β, c, or a_inf unusable).
    """
    c = const.c_ms_per_bit
    if beta is None or beta <= 0 or c <= 0 or a_inf is None or (1.0 + a_inf) <= 0:
        return None
    b = b0
    for _ in range(max(1, iters)):
        bracket = _LN2 * (const.t_ms + const.d0_ms + c * b) + c
        arg = beta * bracket / (c * (1.0 + a_inf))
        if arg <= 0 or not math.isfinite(arg):
            return None
        b = math.log(arg) / _LN2
        if not math.isfinite(b):
            return None
    return b


def _continuous_bstar_lambertw(a_inf, beta, const: ServeConstants):
    """Closed form via scipy's Lambert-W (the W_{-1} branch), if scipy is present.

    Independent cross-check on the fixed point. Returns None if scipy is absent
    or the transform over/underflows (e.g. the d0-dominated degenerate regime,
    where the continuous optimum sits far outside the menu anyway).
    """
    try:
        from scipy.special import lambertw  # noqa: PLC0415
    except Exception:
        return None
    c = const.c_ms_per_bit
    if beta is None or beta <= 0 or c <= 0 or a_inf is None or (1.0 + a_inf) <= 0:
        return None
    try:
        g_over_c = (_LN2 * (const.t_ms + const.d0_ms) + c) / c
        M = (1.0 + a_inf) / (beta * math.exp(g_over_c))
    except OverflowError:
        return None
    if not math.isfinite(M) or M <= 0 or M >= (1.0 / math.e):
        return None  # -M outside [-1/e, 0): no real W_{-1} solution
    w = lambertw(-M, k=-1)
    if abs(w.imag) > 1e-9:
        return None
    s = -w.real
    return s / _LN2 - (const.t_ms + const.d0_ms) / c - 1.0 / _LN2


# --------------------------------------------------------------------------- #
# The selector (doc §3)
# --------------------------------------------------------------------------- #
def select_rung(menu, constants: ServeConstants, accept_points,
                mem_budget_bytes: int, k: int = 1,
                h_source: str = "unknown",
                degenerate_fraction: float = _DEGENERATE_FRACTION,
                ) -> SelectionResult:
    """Pick the throughput-optimal draft rung (doc §3).

    Order of operations:
      1. **Memory gate first** (doc §3.5): keep rungs with
         ``resident_bytes <= mem_budget_bytes``. NOTE: the doc gate is
         ``weights + draft + profiling-peak + 3 GiB margin <= usable pool`` —
         everything except the draft's own resident bytes (weights, profiling
         peak, **and the 3 GiB margin**) is the CALLER's responsibility to net
         out of the usable pool before passing ``mem_budget_bytes``. This
         function compares only the draft footprint against that net budget.
      2. Fit ``(a_inf, β)`` from the served acceptance points.
      3. **Degenerate-regime branch** (doc §3.6): if the cost side varies less
         than ``degenerate_fraction`` of the cycle (``k·c·Δb`` vs ``t + k·d``),
         OR the fit has no usable slope (0/1 acceptance point), the argmax
         provably lands on the acceptance-max rung — pick the **highest-fidelity
         (lowest-E) rung passing the gate** and record ``regime='degenerate'``.
      4. Else **discrete argmax** of ``T(b)`` over the passing rungs (the menu is
         discrete; the continuous Lambert-W optimum lives in provenance as a
         cross-check only).

    Raises ValueError if no rung passes the memory gate (nothing to ship).
    """
    menu = list(menu)
    accept_points = list(accept_points)  # may be iterated several times below
    if not menu:
        raise ValueError("select_rung: empty menu")
    if k < 1:
        raise ValueError(f"select_rung: k must be >= 1, got {k}")

    # 1. Memory gate ---------------------------------------------------------
    passing = [r for r in menu if r.resident_bytes <= mem_budget_bytes]
    excluded = [r for r in menu if r.resident_bytes > mem_budget_bytes]
    if not passing:
        smallest = min(menu, key=lambda r: r.resident_bytes)
        raise ValueError(
            f"select_rung: no rung fits mem_budget_bytes={mem_budget_bytes} "
            f"(smallest is {smallest.name} @ {smallest.resident_bytes} B)")

    # 2. Fit -----------------------------------------------------------------
    E_by_bits = {r.name: r.E for r in menu}
    a_inf, beta, fit_mode = fit_acceptance(accept_points, E_by_bits)

    # per-rung acceptance + a-clamp bookkeeping (only when a_inf is known)
    a_clamped = False
    if a_inf is not None:
        for r in passing:
            raw = a_inf - (beta or 0.0) * math.sqrt(max(r.E, 0.0))
            if raw < 0.0 or raw > 1.0:
                a_clamped = True
                break

    # per-rung throughput (needs a_inf; single_point uses a=a_inf constant)
    per_rung_T = {}
    for r in passing:
        if a_inf is None:
            per_rung_T[r.name] = None
        else:
            a = _acceptance(a_inf, beta, r.E)
            per_rung_T[r.name] = _throughput(a, constants.d(r.bits),
                                             constants.t_ms, k)

    # 3. Degenerate test -----------------------------------------------------
    bits = [r.bits for r in passing]
    b_min, b_max = min(bits), max(bits)
    b_mid = 0.5 * (b_min + b_max)
    cost_span_ms = k * constants.c_ms_per_bit * (b_max - b_min)
    cycle_ms = constants.t_ms + k * constants.d(b_mid)
    ratio = cost_span_ms / cycle_ms if cycle_ms > 0 else 0.0
    cost_flat = ratio < degenerate_fraction
    insufficient_slope = beta is None  # single_point / no_data
    degenerate = cost_flat or insufficient_slope

    if degenerate:
        # Highest fidelity == lowest E among passing rungs. Tie-break toward
        # more bits, then name, for determinism.
        chosen = min(passing, key=lambda r: (r.E, -r.bits, r.name))
        regime = "degenerate"
        reason = "cost_flat" if cost_flat else "insufficient_acceptance_data"
    else:
        # 4. Discrete argmax of T. Tie-break toward higher fidelity (lower E).
        chosen = max(passing, key=lambda r: (per_rung_T[r.name], -r.E, r.name))
        regime = "interior"
        reason = None

    b_star = _continuous_bstar_fixed_point(a_inf, beta, constants, b_mid)
    b_star_lw = _continuous_bstar_lambertw(a_inf, beta, constants)

    provenance = {
        "schema": "mtp_rung_selection/1",
        "selected_rung": chosen.name,
        "selected_bits": chosen.bits,
        "regime": regime,
        "degenerate_reason": reason,
        "k": k,
        "h_source": h_source,
        "constants": {
            "t_ms": constants.t_ms,
            "d0_ms": constants.d0_ms,
            "c_ms_per_bit": constants.c_ms_per_bit,
        },
        "fit": {
            "a_inf": a_inf,
            "beta": beta,
            "fit_mode": fit_mode,
            "n_points": len(list(accept_points)),
            "beta_negative": (beta is not None and beta < 0),
            "points": [
                {"rung": p.rung_name, "bits": p.bits,
                 "E": E_by_bits.get(p.key),
                 "sqrt_E": (math.sqrt(E_by_bits[p.key])
                            if p.key in E_by_bits else None),
                 "acceptance": p.measured_acceptance}
                for p in accept_points
            ],
        },
        "memory": {
            "mem_budget_bytes": int(mem_budget_bytes),
            "passing": [r.name for r in passing],
            "excluded": [{"name": r.name, "resident_bytes": r.resident_bytes}
                         for r in excluded],
        },
        "menu": [
            {"name": r.name, "bits": r.bits, "resident_bytes": r.resident_bytes,
             "E": r.E, "passes_gate": r.resident_bytes <= mem_budget_bytes}
            for r in menu
        ],
        "degenerate_test": {
            "cost_span_ms": cost_span_ms,
            "cycle_ms": cycle_ms,
            "ratio": ratio,
            "threshold": degenerate_fraction,
            "cost_flat": cost_flat,
            "insufficient_slope": insufficient_slope,
            "b_min": b_min,
            "b_max": b_max,
        },
        "per_rung_T": per_rung_T,
        "continuous_bstar": b_star,
        "continuous_method": "fixed_point" if b_star is not None else None,
        "continuous_bstar_lambertw": b_star_lw,
        "a_clamped": a_clamped,
    }
    # Provenance must be JSON-serialisable (doc §3.7); fail fast if not.
    json.dumps(provenance)
    return SelectionResult(rung=chosen, regime=regime, per_rung_T=per_rung_T,
                           provenance=provenance)


# --------------------------------------------------------------------------- #
# Direct measured selector for block-parallel drafts (DSpark and peers)
# --------------------------------------------------------------------------- #
def _configuration_metrics(
    candidate: MeasuredDraftConfiguration,
    ledger: DraftMemoryLedger,
    expected_cycles: Optional[int],
) -> dict:
    expected_tokens = candidate.expected_tokens_per_cycle
    steady_tps = candidate.steady_state_tokens_per_second
    if expected_cycles is None:
        objective_tps = steady_tps
    else:
        objective_tps = (
            1000.0 * expected_cycles * expected_tokens
            / (candidate.load_ms + expected_cycles * candidate.cycle_ms)
        )
    total_peak = ledger.fixed_bytes + candidate.candidate_peak_bytes
    return {
        "k": candidate.k,
        "bits": candidate.bits,
        "E": candidate.E,
        "resident_bytes": candidate.resident_bytes,
        "peak_scratch_bytes": candidate.peak_scratch_bytes,
        "candidate_peak_bytes": candidate.candidate_peak_bytes,
        "total_peak_bytes": total_peak,
        "memory_headroom_bytes": ledger.usable_pool_bytes - total_peak,
        "passes_memory": total_peak <= ledger.usable_pool_bytes,
        "cycle_ms": candidate.cycle_ms,
        "load_ms": candidate.load_ms,
        "position_survival": list(candidate.position_survival),
        "expected_accepted_draft_tokens_per_cycle": math.fsum(
            candidate.position_survival
        ),
        "expected_tokens_per_cycle": expected_tokens,
        "steady_state_tokens_per_second": steady_tps,
        "objective_tokens_per_second": objective_tps,
        "measurement_id": candidate.measurement_id,
        "measurement_source": candidate.measurement_source,
    }


def _pareto_frontier(candidates, metrics: Mapping[str, dict]) -> tuple[str, ...]:
    """Return non-dominated candidates over speed, residency, and load time."""

    candidates = list(candidates)
    frontier: list[str] = []
    for candidate in candidates:
        cm = metrics[candidate.name]
        dominated = False
        for other in candidates:
            if other.name == candidate.name:
                continue
            om = metrics[other.name]
            no_worse = (
                om["steady_state_tokens_per_second"]
                >= cm["steady_state_tokens_per_second"]
                and other.resident_bytes <= candidate.resident_bytes
                and other.load_ms <= candidate.load_ms
            )
            strictly_better = (
                om["steady_state_tokens_per_second"]
                > cm["steady_state_tokens_per_second"]
                or other.resident_bytes < candidate.resident_bytes
                or other.load_ms < candidate.load_ms
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate.name)
    return tuple(sorted(frontier))


def select_measured_configuration(
    menu,
    ledger: DraftMemoryLedger,
    *,
    expected_cycles: Optional[int] = None,
    minimum_k: int = 1,
) -> MeasuredConfigurationSelection:
    """Select a measured block-draft configuration under a hard memory gate.

    A configuration is the joint decision ``(quantization assignment, k)``.
    For candidate ``c`` with cumulative per-position survival ``p[c, i]`` and
    measured complete cycle wall time ``tau[c]``, steady-state throughput is::

        T[c] = 1000 * (1 + sum_i p[c, i]) / tau[c]  tokens/s

    This is the architecture-faithful DSpark objective: all positions are
    generated in one block call, so no ``k*d`` term is introduced.  When
    ``expected_cycles=H`` is supplied, startup is amortized explicitly::

        T_H[c] = 1000 * H * (1 + sum_i p[c, i]) / (load_ms[c] + H*tau[c])

    The hard constraint is the complete :class:`DraftMemoryLedger` plus the
    candidate's resident and peak-scratch bytes.  Candidates below
    ``minimum_k`` (for example DSpark's trained block-size floor) are excluded
    before the argmax.  All candidates must come from the same
    ``measurement_id``; comparing different workloads is a hard error.

    Tie breaks are deterministic and operational: higher steady throughput,
    then lower residency, lower load time, lower measured ``E`` when present,
    and finally lexical name.  The result also reports the non-dominated set
    over steady throughput, resident bytes, and load time.
    """

    menu = list(menu)
    if not menu:
        raise ValueError("select_measured_configuration: empty menu")
    if minimum_k < 1:
        raise ValueError(
            f"select_measured_configuration: minimum_k must be >= 1, got "
            f"{minimum_k}"
        )
    if expected_cycles is not None and (
        not isinstance(expected_cycles, int)
        or isinstance(expected_cycles, bool)
        or expected_cycles < 1
    ):
        raise ValueError(
            "select_measured_configuration: expected_cycles must be a "
            "positive int when present"
        )
    names = [candidate.name for candidate in menu]
    if len(set(names)) != len(names):
        raise ValueError(
            "select_measured_configuration: candidate names must be unique"
        )
    measurement_ids = {candidate.measurement_id for candidate in menu}
    if len(measurement_ids) != 1:
        raise ValueError(
            "select_measured_configuration: candidates use different "
            f"measurement_id values: {sorted(measurement_ids)}"
        )

    per_configuration = {
        candidate.name: _configuration_metrics(
            candidate, ledger, expected_cycles
        )
        for candidate in menu
    }
    for candidate in menu:
        per_configuration[candidate.name]["passes_k_floor"] = (
            candidate.k >= minimum_k
        )

    passing = [
        candidate for candidate in menu
        if candidate.k >= minimum_k
        and per_configuration[candidate.name]["passes_memory"]
    ]
    if not passing:
        smallest = min(
            menu,
            key=lambda candidate: (
                candidate.candidate_peak_bytes, candidate.name
            ),
        )
        raise ValueError(
            "select_measured_configuration: no candidate passes k/memory "
            f"gates (minimum_k={minimum_k}, candidate_budget_bytes="
            f"{ledger.candidate_budget_bytes}; smallest is {smallest.name} @ "
            f"{smallest.candidate_peak_bytes} B)"
        )

    def selection_key(candidate: MeasuredDraftConfiguration):
        metrics = per_configuration[candidate.name]
        E = candidate.E if candidate.E is not None else math.inf
        return (
            -metrics["objective_tokens_per_second"],
            -metrics["steady_state_tokens_per_second"],
            candidate.resident_bytes,
            candidate.load_ms,
            E,
            candidate.name,
        )

    chosen = min(passing, key=selection_key)
    frontier = _pareto_frontier(passing, per_configuration)
    provenance = {
        "schema": "mtp_configuration_selection/1",
        "selected_configuration": chosen.name,
        "objective": (
            "amortized_tokens_per_second"
            if expected_cycles is not None
            else "steady_state_tokens_per_second"
        ),
        "expected_cycles": expected_cycles,
        "minimum_k": minimum_k,
        "measurement_id": next(iter(measurement_ids)),
        "memory": {
            "usable_pool_bytes": ledger.usable_pool_bytes,
            "fixed_runtime_bytes": ledger.fixed_runtime_bytes,
            "target_kv_bytes": ledger.target_kv_bytes,
            "draft_kv_bytes": ledger.draft_kv_bytes,
            "profiling_peak_bytes": ledger.profiling_peak_bytes,
            "safety_margin_bytes": ledger.safety_margin_bytes,
            "fixed_bytes": ledger.fixed_bytes,
            "candidate_budget_bytes": ledger.candidate_budget_bytes,
            "admission_mode": ledger.admission_mode,
        },
        "passing": [candidate.name for candidate in passing],
        "excluded": [
            {
                "name": candidate.name,
                "reasons": [
                    *([] if candidate.k >= minimum_k else ["below_k_floor"]),
                    *([] if per_configuration[candidate.name]["passes_memory"]
                      else ["memory"]),
                ],
            }
            for candidate in menu
            if candidate not in passing
        ],
        "pareto_frontier": list(frontier),
        "per_configuration": per_configuration,
    }
    json.dumps(provenance)
    return MeasuredConfigurationSelection(
        configuration=chosen,
        per_configuration=per_configuration,
        pareto_frontier=frontier,
        provenance=provenance,
    )
