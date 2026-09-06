"""The DP ranks in one currency; a rate-surface anchor set carries its own.

Ported mechanism, not trellis vocabulary (RobTand/prismaquant#127): PR #92's
``trellis_menu._require_run_currency`` refused an enabled seam whose anchors
were weighted SSE under a per-input-channel activation second moment while
the run priced in ``aura-adjoint``. Its successor -- the Tessera campaign --
inherited the same mismatch and lost the refusal: every campaign row stamps
``tessera_campaign.CURRENCY`` and nothing downstream read it, so on the
default ``COST_MODE=aura`` path the DP could rank output-MSE-currency Tessera
rungs against AURA-currency NVFP4/FP8/BF16 rungs in one knapsack.

Three rules, all fail-closed:

1. The COST_MODE -> objective-currency table is derived definitionally from
   the COST_RENDER x COST_OBJECTIVE decomposition ``run-pipeline.sh``
   resolves (``local = inline x weight-recon``,
   ``production-render-score = cached-menu x render-score``,
   ``aura = cached-menu x aura-adjoint``) -- never a threshold anyone picks.
2. The run's objective is read from the ATTESTED ``provenance['cost_mode']``
   the cost stage stamps -- never from ``os.environ``. ``run-pipeline.sh``
   assigns COST_MODE with ``:=`` and never exports it, so an environment read
   in a child process compares the table against a default the run may never
   have used. This module does not import ``os`` at all.
3. An unstamped table carrying Tessera rows is refused rather than compared
   against a default, and a COST_MODE naming no objective is refused rather
   than defaulted.

Every usable Tessera-format row must carry the campaign's currency stamp;
dropping that stamp cannot remove a price from this gate's jurisdiction.
Tables with neither a Tessera format nor a Tessera-currency row pass through
untouched: legacy unstamped stock tables keep their behavior, and other gates
own those rows. Diagnostic error rows are not prices.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: COST_MODE -> the objective currency the run's DP ranks in, read off the
#: COST_RENDER x COST_OBJECTIVE decomposition `run-pipeline.sh` resolves
#: (re-vet R3 block). `production-render` is the legacy spelling of the same
#: objective the driver groups in one case arm; anything else is refused
#: rather than defaulted.
COST_MODE_OBJECTIVE_CURRENCY = {
    "local": "weight-recon",
    "production-render-score": "render-score",
    "production-render": "render-score",
    "aura": "aura-adjoint",
}

#: The objective a Tessera campaign table prices. The campaign measures
#: `output_mse` under the route's activation contract via the production
#: scorer -- the render-score objective -- so this is what its payload stamps
#: into `provenance['cost_mode']`.
RENDER_SCORE_COST_MODE = "production-render-score"


class CostCurrencyError(RuntimeError):
    """A cost table cannot be ranked in this run's objective currency."""


def tessera_campaign_currency() -> str:
    """The currency string the Tessera campaign stamps, read from the module
    that stamps it rather than restated here."""
    from .tessera_campaign import CURRENCY

    return str(CURRENCY)


def _tessera_rows(costs: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(unit, format)`` pairs priced in the Tessera campaign currency."""
    from .tessera_formats import parse_tessera_format_name

    if not isinstance(costs, Mapping):
        return []
    wanted = tessera_campaign_currency()
    found: list[tuple[str, str]] = []
    membership: dict[str, bool] = {}
    for unit, rows in costs.items():
        if not isinstance(rows, Mapping):
            continue
        for fmt, entry in rows.items():
            if not isinstance(entry, Mapping) or "error" in entry:
                continue
            if fmt not in membership:
                try:
                    membership[fmt] = parse_tessera_format_name(fmt) is not None
                except ValueError as exc:
                    raise CostCurrencyError(f"cost row {unit}/{fmt}: {exc}") from exc
            if membership[fmt] and entry.get("currency") != wanted:
                raise CostCurrencyError(
                    f"Tessera cost row {unit}/{fmt} has missing or unknown "
                    f"currency={entry.get('currency')!r}; its producer must "
                    f"stamp the measured campaign currency {wanted!r}")
            if entry.get("currency") == wanted:
                found.append((str(unit), str(fmt)))
    return found


def require_run_currency(cost_data: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a cost table whose Tessera rows are not in the run's currency.

    Takes the whole cost payload (it needs the attested
    ``provenance['cost_mode']`` beside the rows). Returns a report
    ``{cost_mode, expected_currency, tessera_rows}`` so callers can stamp or
    log what was admitted. Never reads ``os.environ``.
    """
    if not isinstance(cost_data, Mapping):
        raise CostCurrencyError("cost payload is not a mapping")
    costs = cost_data.get("costs")
    if not isinstance(costs, Mapping):
        raise CostCurrencyError("cost payload carries no 'costs' table")
    joint = _require_joint_run_currency(cost_data, costs)
    if joint is not None:
        return joint
    tessera = _tessera_rows(costs)
    provenance = cost_data.get("provenance")
    cost_mode = (
        provenance.get("cost_mode")
        if isinstance(provenance, Mapping) else None
    )
    if not tessera:
        return {
            "cost_mode": cost_mode,
            "expected_currency": (
                COST_MODE_OBJECTIVE_CURRENCY.get(cost_mode)
                if isinstance(cost_mode, str) else None
            ),
            "tessera_rows": 0,
        }
    if not isinstance(cost_mode, str) or not cost_mode:
        raise CostCurrencyError(
            f"cost table carries {len(tessera)} Tessera-currency rows "
            f"({tessera_campaign_currency()!r}) but stamps no "
            f"provenance['cost_mode'], so the objective they were measured "
            f"under is unattested (e.g. {tessera[0][0]}/{tessera[0][1]}). An "
            f"unstamped table is refused rather than ranked against a "
            f"default: stamp the COST_MODE the campaign measured under.")
    expected = COST_MODE_OBJECTIVE_CURRENCY.get(cost_mode)
    if expected is None:
        raise CostCurrencyError(
            f"cost table is stamped COST_MODE={cost_mode!r}, which names no "
            f"objective in the COST_RENDER x COST_OBJECTIVE decomposition "
            f"({sorted(COST_MODE_OBJECTIVE_CURRENCY)}). A mode with no "
            f"objective has no currency to compare "
            f"{len(tessera)} Tessera-currency rows against; refusing rather "
            f"than defaulting.")
    if expected != "render-score":
        raise CostCurrencyError(
            f"cost table stamped COST_MODE={cost_mode!r} ranks in "
            f"{expected!r} but carries {len(tessera)} Tessera-currency rows "
            f"priced in {tessera_campaign_currency()!r} (e.g. "
            f"{tessera[0][0]}/{tessera[0][1]}): two numbers in one knapsack "
            f"that are not the same kind of quantity. Price the campaign "
            f"under COST_MODE={RENDER_SCORE_COST_MODE} (the objective its "
            f"output_mse rows measure), or price this run without Tessera "
            f"rows.")
    return {
        "cost_mode": cost_mode,
        "expected_currency": expected,
        "tessera_rows": len(tessera),
    }


def _require_joint_run_currency(cost_data, costs):
    """Joint AURA is an explicitly attested homogeneous measurement table.

    A weight-only AURA row and a joint row use related quadratic objectives,
    but mixing them makes the unmeasured activation side look free. Require
    every usable row, including the BF16 control, to carry the same complete
    joint measurement and probe identity before the allocator sees it.
    """
    provenance = cost_data.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    rows = [(unit, fmt, entry) for unit, per_unit in costs.items()
            if isinstance(per_unit, Mapping)
            for fmt, entry in per_unit.items()
            if isinstance(entry, Mapping) and "error" not in entry]
    claimed = provenance.get("joint_activation") is True or any(
        entry.get("cost_source") == "joint_aura"
        or entry.get("cost_currency") == "joint_aura_predicted_dloss"
        or "joint_operator_identity" in entry
        or "joint_operator_identity_sha256" in entry
        for _, _, entry in rows)
    if not claimed:
        return None
    from .joint_aura import JOINT_AURA_COST_CURRENCY, validate_joint_aura_entry
    from .tessera_formats import parse_tessera_format_name

    if (provenance.get("cost_mode") != "aura"
            or provenance.get("joint_activation") is not True
            or provenance.get("cost_currency") != JOINT_AURA_COST_CURRENCY):
        raise CostCurrencyError("joint AURA requires matching aura/joint provenance")
    if not rows:
        raise CostCurrencyError("joint AURA table has no measured rows")
    probe_identity = None
    tessera_count = 0
    for unit, fmt, entry in rows:
        try:
            if not validate_joint_aura_entry(entry):
                raise ValueError("mixes joint and non-joint cost rows")
            operator = entry["joint_operator_identity"]
            if operator["qname"] != unit or operator["format"] != fmt:
                raise ValueError("operator identity differs from its cost-table key")
            current = entry["probe_identity_sha256"]
            if probe_identity is not None and current != probe_identity:
                raise ValueError("rows do not share one probe/calibration identity")
            probe_identity = current
            tessera_count += parse_tessera_format_name(fmt) is not None
        except (ValueError, TypeError, KeyError) as exc:
            raise CostCurrencyError(f"joint AURA cost row {unit}/{fmt}: {exc}") from exc
    return {"cost_mode": "aura", "expected_currency": "aura-adjoint",
            "cost_currency": JOINT_AURA_COST_CURRENCY,
            "tessera_rows": tessera_count, "joint_aura_rows": len(rows),
            "probe_identity_sha256": probe_identity,
            "activation_quantization_included": True,
            "measurement_status": "research"}
