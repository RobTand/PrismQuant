"""Format-neutral shared log shapes and sparse per-unit anchor corrections.

All logarithms are base ten. Callers own currency, equivalence segments,
measurement identity, validation policy and fallback. A shared panel removes
per-unit log offsets before fitting its feature basis. One anchor restores a
unit's level; two or more also fit a residual slope in a centered, scaled
coordinate. Real anchors remain exact, including in an overdetermined fit.

Audit measurements never refit the model. An explicit later fit may consume
accepted audit measurements without changing the frozen earlier predictions.
This module performs scalar offline arithmetic and imports no model runtime.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType


class AnchoredShapeError(ValueError):
    """Invalid or unrepresentable numerical shape input."""


def _label(value: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnchoredShapeError(f"{where} must be a nonempty string")
    return value


def _finite(value: float, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise AnchoredShapeError(f"{where} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnchoredShapeError(f"{where} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise AnchoredShapeError(f"{where} must be finite" + (" and positive" if positive else ""))
    return result


@dataclass(frozen=True)
class LogShapeObservation:
    unit: str
    key: str
    value: float

    def __post_init__(self) -> None:
        _label(self.unit, "observation unit")
        _label(self.key, "observation key")
        object.__setattr__(self, "value", _finite(self.value, "observation value", positive=True))


@dataclass(frozen=True)
class SharedLogShape:
    coefficients: tuple[float, ...]
    reference_key: str
    log_shape_by_key: Mapping[str, float]
    design_rank: int
    n_units: int
    n_observations: int

    def __post_init__(self) -> None:
        values = {_label(key, "shape key"): _finite(value, "log shape")
                  for key, value in self.log_shape_by_key.items()}
        coefficients = tuple(_finite(value, "shape coefficient") for value in self.coefficients)
        if not values or self.reference_key not in values:
            raise AnchoredShapeError("shape reference is absent")
        if not coefficients or self.design_rank != len(coefficients):
            raise AnchoredShapeError("shape design rank is insufficient")
        if self.n_units < 1 or self.n_observations < 2*self.n_units:
            raise AnchoredShapeError("shape observation count is insufficient")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "log_shape_by_key", MappingProxyType(values))


@dataclass(frozen=True)
class AnchorCorrection:
    shape: SharedLogShape
    anchors: Mapping[str, float]
    coordinates: Mapping[str, float]
    center: float
    scale: float
    intercept: float
    slope: float

    def __post_init__(self) -> None:
        # Freeze caller-owned mappings so later measurement/refit cannot
        # change predictions used to evaluate an earlier audit.
        if not isinstance(self.shape, SharedLogShape):
            raise AnchoredShapeError("shared shape type is invalid")
        anchors = {_label(key, "anchor key"): _finite(value, "anchor value", positive=True)
                   for key, value in self.anchors.items()}
        coordinates = {_label(key, "coordinate key"): _finite(value, "coordinate")
                       for key, value in self.coordinates.items()}
        if not anchors or not set(anchors) <= set(coordinates):
            raise AnchoredShapeError("anchor keys are empty or unknown")
        if set(coordinates) != set(self.shape.log_shape_by_key):
            raise AnchoredShapeError("coordinate domain differs from shared shape")
        if len(set(coordinates.values())) != len(coordinates):
            raise AnchoredShapeError("duplicate coordinates do not identify distinct rungs")
        for name in ("center", "scale", "intercept", "slope"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=name == "scale"))
        object.__setattr__(self, "anchors", MappingProxyType(anchors))
        object.__setattr__(self, "coordinates", MappingProxyType(coordinates))


@dataclass(frozen=True)
class AnchoredAuditRow:
    key: str
    predicted: float
    measured: float
    absolute_log10_error: float


@dataclass(frozen=True)
class AnchoredAudit:
    rows: tuple[AnchoredAuditRow, ...]

    @property
    def max_absolute_log10_error(self) -> float:
        return max(row.absolute_log10_error for row in self.rows)


def rank_and_solve(
    x_rows: Sequence[Sequence[float]],
    y_rows: Sequence[float],
) -> tuple[int, tuple[float, ...]]:
    """Solve centered least squares through normal equations with pivoting."""
    if not x_rows:
        raise AnchoredShapeError("shape design is empty")
    width = len(x_rows[0])
    if width < 1 or any(len(row) != width for row in x_rows):
        raise AnchoredShapeError("shape feature width is invalid")
    # Plugin features have no prescribed unit.  Normalize every column before
    # rank detection/solving so a plugin expressing the same coordinate in
    # 1e-7 rather than 1 cannot turn an identifiable design into rank zero.
    column_scales = [
        math.sqrt(math.fsum(row[index] * row[index] for row in x_rows))
        for index in range(width)
    ]
    active = [scale > 0.0 and math.isfinite(scale) for scale in column_scales]
    if not all(active):
        return sum(active), tuple()
    normalized = [
        [row[index] / column_scales[index] for index in range(width)]
        for row in x_rows
    ]
    gram = [
        [math.fsum(row[i] * row[j] for row in normalized) for j in range(width)]
        for i in range(width)
    ]
    rhs = [
        math.fsum(row[i] * value for row, value in zip(normalized, y_rows))
        for i in range(width)
    ]
    scale = max((abs(value) for row in gram for value in row), default=1.0)
    tolerance = scale * 1e-12
    augmented = [gram[index] + [rhs[index]] for index in range(width)]
    rank = 0
    for column in range(width):
        pivot = max(
            range(rank, width), key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) <= tolerance:
            continue
        augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]
        divisor = augmented[rank][column]
        augmented[rank] = [value / divisor for value in augmented[rank]]
        for row in range(width):
            if row == rank:
                continue
            factor = augmented[row][column]
            if abs(factor) <= tolerance:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[rank],
                )
            ]
        rank += 1
    if rank != width:
        return rank, tuple()
    solution = [0.0] * width
    for row in range(width):
        pivot_columns = [
            column for column in range(width)
            if abs(augmented[row][column] - 1.0) <= 1e-9
            and all(
                abs(augmented[other][column]) <= 1e-9
                for other in range(width) if other != row
            )
        ]
        if len(pivot_columns) != 1:
            raise AnchoredShapeError("shape solve pivot reconstruction failed")
        solution[pivot_columns[0]] = augmented[row][-1]
    return rank, tuple(
        solution[index] / column_scales[index] for index in range(width)
    )


def fit_centered_log_shape(
    observations: Sequence[LogShapeObservation],
    features_by_key: Mapping[str, Sequence[float]],
    *,
    reference_key: str | None = None,
) -> SharedLogShape:
    """Fit log shape after removing each panel unit's own log-cost level.

    A caller may declare unmeasured target keys, provided the observed panel
    identifies every feature column. Feature and observation iteration order
    is retained to preserve the established AURA fitter's arithmetic.
    """
    if not observations or not features_by_key:
        raise AnchoredShapeError("shape input is empty")
    features = {
        _label(key, "feature key"): tuple(_finite(value, "shape feature") for value in row)
        for key, row in features_by_key.items()
    }
    widths = {len(row) for row in features.values()}
    if len(widths) != 1 or not next(iter(widths)):
        raise AnchoredShapeError("shape feature width is invalid")
    width = next(iter(widths))
    reference = min(features) if reference_key is None else reference_key
    if reference not in features:
        raise AnchoredShapeError("unknown shape reference key")
    by_unit: dict[str, list[tuple[tuple[float, ...], float]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    duplicate = False
    for observation in observations:
        if not isinstance(observation, LogShapeObservation):
            raise AnchoredShapeError("shape observation type is invalid")
        pair = observation.unit, observation.key
        if pair in seen:
            duplicate = True
        seen.add(pair)
        if observation.key not in features:
            raise AnchoredShapeError("unknown shape observation key")
        by_unit[observation.unit].append((features[observation.key], math.log10(observation.value)))
    x_rows: list[tuple[float, ...]] = []
    y_rows: list[float] = []
    try:
        for unit, rows in sorted(by_unit.items()):
            if len(rows) < 2:
                raise AnchoredShapeError(f"panel unit {unit!r} has fewer than two rungs")
            feature_means = tuple(math.fsum(row[index] for row, _ in rows) / len(rows)
                                  for index in range(width))
            value_mean = math.fsum(value for _, value in rows) / len(rows)
            for row, value in rows:
                x_rows.append(tuple(row[index] - feature_means[index] for index in range(width)))
                y_rows.append(value - value_mean)
        rank, coefficients = rank_and_solve(x_rows, y_rows)
        if rank != width:
            raise AnchoredShapeError(f"panel design rank is {rank} of {width}")
        if duplicate:
            raise AnchoredShapeError("duplicate unit/key shape observation")
        log_shape = {
            key: math.fsum(coefficient * (feature - features[reference][index])
                           for index, (coefficient, feature) in enumerate(zip(coefficients, row)))
            for key, row in features.items()
        }
    except (OverflowError, ZeroDivisionError) as exc:
        raise AnchoredShapeError("shape design arithmetic is not representable") from exc
    return SharedLogShape(coefficients, reference, log_shape, rank, len(by_unit), len(observations))


def fit_anchor_correction(
    shape: SharedLogShape,
    anchors: Mapping[str, float],
    coordinates: Mapping[str, float],
) -> AnchorCorrection:
    """Restore level from one anchor, or fit level and tilt from two or more.

    The coordinate map declares the entire supported target domain. Center
    and scale it before solving, so coordinate origin and units cannot create
    an artificial rank failure. Beyond two anchors the affine residual is a
    least-squares proposal; each measured anchor still predicts exactly.
    """
    if not isinstance(shape, SharedLogShape):
        raise AnchoredShapeError("shared shape type is invalid")
    if not anchors:
        raise AnchoredShapeError("anchor input is empty")
    if set(coordinates) != set(shape.log_shape_by_key):
        raise AnchoredShapeError("coordinate domain differs from shared shape")
    coords = {key: _finite(value, "coordinate") for key, value in coordinates.items()}
    if len(set(coords.values())) != len(coords):
        raise AnchoredShapeError("duplicate coordinates do not identify distinct rungs")
    values: dict[str, float] = {}
    for key, value in anchors.items():
        if key not in coords:
            raise AnchoredShapeError("unknown anchor key")
        values[key] = _finite(value, "anchor value", positive=True)
    lower, upper = min(coords.values()), max(coords.values())
    center = lower/2.0 + upper/2.0
    scale = max(abs(lower - center), abs(upper - center)) or 1.0
    normalized = {key: (value - center)/scale for key, value in coords.items()}
    if any(not math.isfinite(value) for value in normalized.values()):
        raise AnchoredShapeError("coordinate normalization is not representable")
    if len(set(normalized.values())) != len(coords):
        raise AnchoredShapeError("coordinate normalization loses distinct rungs")
    xs = [normalized[key] for key in values]
    residuals = [_finite(math.log10(value) - shape.log_shape_by_key[key], "anchor residual")
                 for key, value in values.items()]
    if len(values) == 1:
        intercept, slope = residuals[0], 0.0
    else:
        try:
            x_mean = math.fsum(xs)/len(xs)
            y_mean = math.fsum(residuals)/len(residuals)
            rank, coefficients = rank_and_solve([(x-x_mean,) for x in xs],
                                                [y-y_mean for y in residuals])
        except (OverflowError, ZeroDivisionError) as exc:
            raise AnchoredShapeError("anchor correction arithmetic is not representable") from exc
        if rank != 1:
            raise AnchoredShapeError("anchor correction design rank is insufficient")
        slope = coefficients[0]
        intercept = y_mean - slope*x_mean
    return AnchorCorrection(shape, values, coords, center, scale,
                            _finite(intercept, "residual intercept"),
                            _finite(slope, "residual slope"))


def predict_anchored(correction: AnchorCorrection, key: str) -> float:
    """Predict one declared key, returning actual anchor values verbatim."""
    if not isinstance(correction, AnchorCorrection):
        raise AnchoredShapeError("anchor correction type is invalid")
    _label(key, "prediction key")
    if key not in correction.coordinates:
        raise AnchoredShapeError("unknown prediction key")
    if key in correction.anchors:
        return correction.anchors[key]
    coordinate = (correction.coordinates[key] - correction.center)/correction.scale
    try:
        log_value = math.fsum((correction.shape.log_shape_by_key[key],
                               correction.intercept, correction.slope*coordinate))
        value = 10.0**log_value
    except (OverflowError, ValueError) as exc:
        raise AnchoredShapeError("prediction is not representable as a positive finite float") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise AnchoredShapeError("prediction is not representable as a positive finite float")
    return value


def audit_anchored(
    correction: AnchorCorrection,
    observations: Mapping[str, float],
) -> AnchoredAudit:
    """Freeze untouched audit predictions and errors without fitting them."""
    if not observations:
        raise AnchoredShapeError("audit input is empty")
    if set(observations) & set(correction.anchors):
        raise AnchoredShapeError("audit key is also a fitted anchor")
    rows = []
    for key, value in sorted(observations.items()):
        measured = _finite(value, "audit value", positive=True)
        predicted = predict_anchored(correction, key)
        rows.append(AnchoredAuditRow(key, predicted, measured,
                                    abs(math.log10(predicted) - math.log10(measured))))
    return AnchoredAudit(tuple(rows))


__all__ = [
    "AnchoredShapeError", "LogShapeObservation", "SharedLogShape", "AnchorCorrection",
    "AnchoredAuditRow", "AnchoredAudit", "fit_centered_log_shape", "fit_anchor_correction",
    "predict_anchored", "audit_anchored",
]
