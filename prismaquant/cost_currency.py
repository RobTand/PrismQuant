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

Tables with no Tessera-currency row are outside this gate's jurisdiction and
pass through untouched: legacy unstamped stock tables keep their behavior,
and other gates own those rows.
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
    if not isinstance(costs, Mapping):
        return []
    wanted = tessera_campaign_currency()
    found: list[tuple[str, str]] = []
    for unit, rows in costs.items():
        if not isinstance(rows, Mapping):
            continue
        for fmt, entry in rows.items():
            if isinstance(entry, Mapping) and entry.get("currency") == wanted:
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
