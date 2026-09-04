"""Runtime schema checks for PrismaQuant file handoffs.

The pipeline passes several pickle/JSON artifacts between long-running
steps.  These validators intentionally check only the structural contract
that downstream code relies on, so older artifacts with extra fields still
load while malformed artifacts fail before optimization or export begins.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import NotRequired, TypedDict


class CostEntry(TypedDict, total=False):
    """Structural type for one per-format cost row.

    Older artifacts are allowed to omit ``cost_source``; live producers should
    set it when a cost was rewritten or comes from a non-default surrogate so
    allocator logs can explain which objective priced the decision.
    """

    weight_mse: float
    output_mse: float
    rel_output_mse: float
    predicted_dloss: float
    fisher_output_mse: float
    output_mse_measured: bool
    cost_source: NotRequired[str]
    weight_mse_per_expert: NotRequired[list[float]]
    cost_source_per_expert: NotRequired[list[str]]
    cb_minchain_identity_per_expert: NotRequired[list[dict]]
    cb_minchain_interpolation: NotRequired[dict]
    error: str


class SchemaValidationError(ValueError):
    """Raised when a PrismaQuant handoff artifact is structurally invalid."""


def _label(path: str | None) -> str:
    return str(path) if path else "<memory>"


def _fail(path: str | None, where: str, message: str) -> None:
    raise SchemaValidationError(f"{_label(path)}:{where}: {message}")


def _is_mapping(value) -> bool:
    return isinstance(value, Mapping)


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _as_non_negative_int(value, path: str | None, where: str) -> int:
    if isinstance(value, bool):
        _fail(path, where, "expected a non-negative integer")
    try:
        out = int(value)
    except (TypeError, ValueError):
        _fail(path, where, "expected a non-negative integer")
    if out < 0:
        _fail(path, where, "expected a non-negative integer")
    return out


def _as_number(value, path: str | None, where: str) -> float:
    if not _is_number(value):
        _fail(path, where, "expected a number")
    return float(value)


def _as_finite_cost_number(value, path: str | None, where: str) -> float:
    """A usable cost must be finite; other handoff schemas keep their rules."""
    out = _as_number(value, path, where)
    if not math.isfinite(out):
        _fail(path, where, "expected a finite number")
    return out


def _validate_router_number_map(
    payload,
    field: str,
    path: str | None,
    *,
    integral_values: bool = False,
) -> None:
    value = payload.get(field)
    if value is None:
        return
    if not _is_mapping(value):
        _fail(path, f".{field}", "must be a mapping when present")
    for router, values in value.items():
        if not isinstance(router, str):
            _fail(path, f".{field}", "router keys must be strings")
        if not _is_mapping(values):
            _fail(path, f".{field}[{router!r}]", "must be a mapping")
        for eid, count in values.items():
            if not isinstance(eid, (str, Integral)) or isinstance(eid, bool):
                _fail(path, f".{field}[{router!r}]", "expert ids must be strings or ints")
            if integral_values:
                _as_non_negative_int(count, path, f".{field}[{router!r}][{eid!r}]")
            else:
                _as_number(count, path, f".{field}[{router!r}][{eid!r}]")


def validate_probe_payload(payload, path: str | None = None):
    """Validate the merged sensitivity-probe pickle contract."""
    if not _is_mapping(payload):
        _fail(path, "", "probe payload is not a mapping")
    stats = payload.get("stats")
    if not _is_mapping(stats):
        _fail(path, ".stats", "missing or not a mapping")
    for name, entry in stats.items():
        if not isinstance(name, str):
            _fail(path, ".stats", "stat keys must be strings")
        if not _is_mapping(entry):
            _fail(path, f".stats[{name!r}]", "entry is not a mapping")
        if "h_trace" not in entry:
            _fail(path, f".stats[{name!r}].h_trace", "required field missing")
        if "n_params" not in entry:
            _fail(path, f".stats[{name!r}].n_params", "required field missing")
        _as_number(entry["h_trace"], path, f".stats[{name!r}].h_trace")
        _as_non_negative_int(entry["n_params"], path, f".stats[{name!r}].n_params")
        for optional in ("in_features", "out_features", "num_experts"):
            if optional in entry and entry[optional] is not None:
                _as_non_negative_int(
                    entry[optional], path, f".stats[{name!r}].{optional}"
                )
    meta = payload.get("meta", {})
    if meta is not None and not _is_mapping(meta):
        _fail(path, ".meta", "must be a mapping when present")
    _validate_router_number_map(payload, "router_counts", path)
    _validate_router_number_map(payload, "router_active_counts", path, integral_values=True)
    router_totals = payload.get("router_totals")
    if router_totals is not None:
        if not _is_mapping(router_totals):
            _fail(path, ".router_totals", "must be a mapping when present")
        for router, total in router_totals.items():
            if not isinstance(router, str):
                _fail(path, ".router_totals", "router keys must be strings")
            _as_non_negative_int(total, path, f".router_totals[{router!r}]")
    expert_info = payload.get("expert_info", {})
    if expert_info is not None:
        if not _is_mapping(expert_info):
            _fail(path, ".expert_info", "must be a mapping when present")
        for name, pair in expert_info.items():
            if not isinstance(name, str):
                _fail(path, ".expert_info", "expert-info keys must be strings")
            if (not isinstance(pair, Sequence)
                    or isinstance(pair, (str, bytes))
                    or len(pair) != 2):
                _fail(path, f".expert_info[{name!r}]", "must be a 2-item sequence")
            router, eid = pair
            if not isinstance(router, str):
                _fail(path, f".expert_info[{name!r}][0]", "router qname must be a string")
            if not isinstance(eid, (str, Integral)) or isinstance(eid, bool):
                _fail(path, f".expert_info[{name!r}][1]", "expert id must be a string or int")
    return payload


def validate_cost_payload(payload, path: str | None = None):
    """Validate cost structure and finite numeric signals on non-error rows."""
    if not _is_mapping(payload):
        _fail(path, "", "cost payload is not a mapping")
    costs = payload.get("costs")
    if not _is_mapping(costs):
        _fail(path, ".costs", "missing or not a mapping")
    formats = payload.get("formats", [])
    if formats is not None:
        if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
            _fail(path, ".formats", "must be a sequence of format names")
        for idx, fmt in enumerate(formats):
            if not isinstance(fmt, str):
                _fail(path, f".formats[{idx}]", "format name must be a string")
    for name, layer_costs in costs.items():
        if not isinstance(name, str):
            _fail(path, ".costs", "layer keys must be strings")
        if not _is_mapping(layer_costs):
            _fail(path, f".costs[{name!r}]", "entry is not a mapping")
        for fmt, entry in layer_costs.items():
            if not isinstance(fmt, str):
                _fail(path, f".costs[{name!r}]", "format keys must be strings")
            if not _is_mapping(entry):
                _fail(path, f".costs[{name!r}][{fmt!r}]", "entry is not a mapping")
            if "error" in entry:
                continue
            has_signal = False
            for field in (
                "weight_mse",
                "predicted_dloss",
                "output_mse",
                "fisher_output_mse",
            ):
                if field in entry:
                    _as_finite_cost_number(
                        entry[field], path, f".costs[{name!r}][{fmt!r}].{field}"
                    )
                    has_signal = True
            if "cost_source" in entry and not isinstance(entry["cost_source"], str):
                _fail(
                    path,
                    f".costs[{name!r}][{fmt!r}].cost_source",
                    "must be a string when present",
                )
            if "weight_mse_per_expert" in entry:
                values = entry["weight_mse_per_expert"]
                if (not isinstance(values, Sequence)
                        or isinstance(values, (str, bytes))):
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}].weight_mse_per_expert",
                        "must be a sequence when present",
                    )
                for idx, value in enumerate(values):
                    _as_finite_cost_number(
                        value,
                        path,
                        f".costs[{name!r}][{fmt!r}]"
                        f".weight_mse_per_expert[{idx}]",
                    )
            if "cost_source_per_expert" in entry:
                values = entry["cost_source_per_expert"]
                if (not isinstance(values, Sequence)
                        or isinstance(values, (str, bytes))
                        or not all(isinstance(value, str) for value in values)):
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}].cost_source_per_expert",
                        "must be a sequence of strings when present",
                    )
                mse_values = entry.get("weight_mse_per_expert")
                if (isinstance(mse_values, Sequence)
                        and not isinstance(mse_values, (str, bytes))
                        and len(values) != len(mse_values)):
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}].cost_source_per_expert",
                        "must match weight_mse_per_expert length",
                    )
            if "cb_minchain_identity_per_expert" in entry:
                identities = entry["cb_minchain_identity_per_expert"]
                if (not isinstance(identities, Sequence)
                        or isinstance(identities, (str, bytes))):
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}]"
                        ".cb_minchain_identity_per_expert",
                        "must be a sequence when present",
                    )
                from .cb_minchain import validate_chain_identity

                for idx, identity in enumerate(identities):
                    try:
                        validate_chain_identity(
                            identity,
                            where=(
                                f".costs[{name!r}][{fmt!r}]"
                                f".cb_minchain_identity_per_expert[{idx}]"
                            ),
                        )
                    except ValueError as exc:
                        _fail(path, "", str(exc))
                mse_values = entry.get("weight_mse_per_expert")
                if (isinstance(mse_values, Sequence)
                        and not isinstance(mse_values, (str, bytes))
                        and len(identities) != len(mse_values)):
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}]"
                        ".cb_minchain_identity_per_expert",
                        "must match weight_mse_per_expert length",
                    )
            if "cb_minchain_interpolation" in entry:
                interpolation = entry["cb_minchain_interpolation"]
                if not _is_mapping(interpolation):
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}]"
                        ".cb_minchain_interpolation",
                        "must be an object when present",
                    )
                if interpolation.get("semantic") != (
                    "v2_accept_all_plus_per_layer_audit"
                ):
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}]"
                        ".cb_minchain_interpolation.semantic",
                        "has an unsupported interpolation semantic",
                    )
                if interpolation.get("layer_audit_pass") is not True:
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}]"
                        ".cb_minchain_interpolation.layer_audit_pass",
                        "must be true for an interpolated row",
                    )
            if ("output_mse_measured" in entry
                    and not isinstance(entry["output_mse_measured"], bool)):
                _fail(
                    path,
                    f".costs[{name!r}][{fmt!r}].output_mse_measured",
                    "must be a boolean when present",
                )
            if not has_signal:
                _fail(
                    path,
                    f".costs[{name!r}][{fmt!r}]",
                    "usable cost entry needs weight_mse, predicted_dloss, or output_mse",
                )
    return payload


def validate_layer_config_payload(payload, path: str | None = None):
    """Validate allocator/exporter layer_config JSON shape."""
    if not _is_mapping(payload):
        _fail(path, "", "layer_config is not a JSON object")
    for name, entry in payload.items():
        if not isinstance(name, str):
            _fail(path, "", "layer_config keys must be strings")
        if name == "__prismaquant__":
            # Reserved allocator-metadata block (layer_config.LAYER_CONFIG_META_KEY):
            # travels with the assignment, is not a tensor entry.
            if not _is_mapping(entry):
                _fail(path, f"[{name!r}]", "reserved metadata must be an object")
            continue
        where = f"[{name!r}]"
        if isinstance(entry, dict):
            dt = entry.get("data_type")
            if not isinstance(dt, str):
                _fail(path, f"{where}.data_type", "required string field missing")
            if "bits" in entry:
                _as_non_negative_int(entry["bits"], path, f"{where}.bits")
            if "group_size" in entry and entry["group_size"] is not None:
                _as_non_negative_int(entry["group_size"], path, f"{where}.group_size")
            continue
        if isinstance(entry, str):
            continue
        if isinstance(entry, int) and not isinstance(entry, bool):
            continue
        _fail(path, where, "entry must be a format dict, string, or integer")
    return payload
