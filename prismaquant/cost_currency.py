"""One DP prices in one currency, and the currency is a value a gate reads.

A cost table's rows are Δloss estimates the knapsack maximises over.  They
are only comparable when they are the same *kind* of quantity, and this tree
produces three kinds, one per ``COST_OBJECTIVE`` (``run-pipeline.sh``,
re-vet R3): a weight-reconstruction error, an output-MSE render score
carried onto the loss scale by ``½·h_trace``, and the AURA KL-adjoint
projection.  ``COST_MODE`` is the documented spelling of
``COST_RENDER x COST_OBJECTIVE``; the objective half is the currency.

Two producers write rows in a currency that is *not* implied by the mode the
run was launched under, and both say so on the row:

* ``tessera_campaign`` stamps ``currency`` -- output MSE measured under the
  route's activation contract, which is ``production-render-score``'s
  ``--score-field output_mse`` and explicitly not the AURA adjoint (its own
  module header says so);
* the anchored-AURA campaigns stamp ``cost_currency`` --
  ``aura_predicted_dloss``, a KL-adjoint projection and nothing else.

Until 2026-09-03 nothing read the first stamp (RobTand/prismaquant#127): a
Tessera campaign table fed to an ``aura`` allocation put Fisher-transferred
output MSE and KL-adjoint projections into one ``max()``.  The retired
trellis seam refused exactly this (``archive/trellis_wire_2026-09-02``,
``_require_run_currency``); its successor kept the stamp and lost the
refusal.  This module is the refusal, ported by mechanism:

1. the run's objective is read from the table's **attested**
   ``provenance["cost_mode"]`` -- never from ``os.environ``, which
   ``run-pipeline.sh`` never exports and which therefore describes whatever
   shell the reader happens to be in, not the run;
2. a stamped mode that names no objective refuses rather than defaulting;
3. a row that declares a currency is compared against that objective, and a
   mismatch refuses;
4. a table whose rows declare a currency and which carries no stamp refuses,
   because there is nothing attested to compare against.

A table with no stamp *and* no declared currency is the pre-R2 legacy
``run-pipeline.sh`` reuses with a warning (``cost_table_reusable``): every
row in it came from one cost stage in that stage's one currency, and there is
nothing on it for this gate to compare.  It passes with the verdict
``undeclared_unstamped`` stamped into the allocation's provenance, so the
artifact records that the check had nothing to read rather than that it
passed.

The tables here are the single source of both spellings: producers import
their currency string from this module, and a test holds
``COST_MODE_OBJECTIVE`` equal to the ``case`` block in ``run-pipeline.sh``.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

__all__ = [
    "ANCHORED_AURA_COST_CURRENCY",
    "COST_MODE_OBJECTIVE",
    "CostCurrencyError",
    "ROW_CURRENCY_FIELDS",
    "ROW_CURRENCY_OBJECTIVE",
    "TESSERA_CAMPAIGN_CURRENCY",
    "VERDICT_SINGLE_CURRENCY",
    "VERDICT_UNDECLARED_UNSTAMPED",
    "cost_mode_for_currency",
    "declared_row_currencies",
    "require_cost_currency",
]

#: ``COST_MODE`` -> ``COST_OBJECTIVE``, the objective half of the
#: ``COST_RENDER x COST_OBJECTIVE`` decomposition ``run-pipeline.sh`` resolves
#: (its ``case "$COST_MODE"`` block).  Mirrored, and pinned equal by
#: ``tests/test_cost_currency.py``; the shell is the authority.
COST_MODE_OBJECTIVE: Mapping[str, str] = {
    "local": "weight-recon",
    "production-render-score": "render-score",
    "production-render": "render-score",
    "aura": "aura-adjoint",
}

#: The Tessera campaign's row currency (``tessera_campaign.CURRENCY``).
TESSERA_CAMPAIGN_CURRENCY = "output_mse_under_route_activation_contract"
#: The anchored-AURA campaigns' row currency
#: (``allocator_candidates.ANCHORED_AURA_COST_CURRENCY``).
ANCHORED_AURA_COST_CURRENCY = "aura_predicted_dloss"

#: Declared row currency -> the objective it is a measurement of.  A currency
#: absent from this table is refused: an unknown unit cannot be compared.
ROW_CURRENCY_OBJECTIVE: Mapping[str, str] = {
    TESSERA_CAMPAIGN_CURRENCY: "render-score",
    ANCHORED_AURA_COST_CURRENCY: "aura-adjoint",
}

#: The row fields a producer declares its currency in.  Two spellings exist
#: because two producers wrote them before anything read either; the gate
#: reads both and a row may not carry two that disagree.
ROW_CURRENCY_FIELDS = ("currency", "cost_currency")

VERDICT_SINGLE_CURRENCY = "single_currency"
VERDICT_UNDECLARED_UNSTAMPED = "undeclared_unstamped"


class CostCurrencyError(ValueError):
    """The cost table cannot be ranked in one currency."""


def cost_mode_for_currency(currency: str) -> str:
    """The canonical ``COST_MODE`` whose objective a row currency measures.

    Used by a producer to *derive* the ``cost_mode`` it stamps from the
    currency it writes, rather than restating a second string.  The first
    mode in ``COST_MODE_OBJECTIVE`` order naming that objective is the
    canonical spelling (``production-render-score`` over its alias).
    """
    try:
        objective = ROW_CURRENCY_OBJECTIVE[currency]
    except KeyError:
        raise CostCurrencyError(
            f"row currency {currency!r} names no cost objective; known: "
            f"{sorted(ROW_CURRENCY_OBJECTIVE)}") from None
    for mode, mode_objective in COST_MODE_OBJECTIVE.items():
        if mode_objective == objective:
            return mode
    raise CostCurrencyError(     # unreachable while the two tables agree
        f"objective {objective!r} is named by no COST_MODE")


def _row_currency(entry: Mapping, *, where: str) -> str | None:
    found = {
        field: entry[field] for field in ROW_CURRENCY_FIELDS
        if isinstance(entry.get(field), str) and entry[field]
    }
    if not found:
        return None
    values = set(found.values())
    if len(values) > 1:
        raise CostCurrencyError(
            f"{where}: row declares two currencies that disagree: {found}")
    return values.pop()


def declared_row_currencies(costs: Mapping, *, where: str = "cost table") -> Counter:
    """``{currency: rows}`` over every row that declares one."""
    declared: Counter = Counter()
    for unit, rows in costs.items():
        if not isinstance(rows, Mapping):
            continue
        for fmt, entry in rows.items():
            if not isinstance(entry, Mapping):
                continue
            currency = _row_currency(entry, where=f"{where} [{unit}][{fmt}]")
            if currency is not None:
                declared[currency] += 1
    return declared


def require_cost_currency(payload: Mapping, *, where: str = "cost table") -> dict:
    """Refuse a cost table the DP cannot rank in one currency.

    Reads the attested ``provenance["cost_mode"]`` and every declared row
    currency (and the table-level ``currency`` the campaign stamps), and
    returns the verdict to stamp into the allocation's provenance.  Raises
    :class:`CostCurrencyError` on a stamped mode naming no objective, on a
    declared currency naming no objective, on a declared currency whose
    objective is not the stamped one, and on a declared currency with no
    stamp to compare it against.  Never consults ``os.environ``.
    """
    provenance = payload.get("provenance") if isinstance(payload, Mapping) else None
    stamped = None
    if isinstance(provenance, Mapping):
        value = provenance.get("cost_mode")
        stamped = str(value) if isinstance(value, str) and value else None

    declared = declared_row_currencies(payload.get("costs", {}) or {}, where=where)
    table_currency = payload.get("currency")
    if isinstance(table_currency, str) and table_currency:
        declared.setdefault(table_currency, 0)

    unknown = sorted(c for c in declared if c not in ROW_CURRENCY_OBJECTIVE)
    if unknown:
        raise CostCurrencyError(
            f"{where}: declared currency {unknown} names no cost objective "
            f"(known: {sorted(ROW_CURRENCY_OBJECTIVE)}); refusing to rank a "
            "quantity in an unknown unit")

    if stamped is None:
        if declared:
            raise CostCurrencyError(
                f"{where}: rows declare currency {dict(declared)} but the table "
                "carries no provenance['cost_mode'] to compare against; an "
                "unstamped table is refused, not compared against a default "
                "(re-vet R2; RobTand/prismaquant#127)")
        return {
            "verdict": VERDICT_UNDECLARED_UNSTAMPED,
            "cost_mode": None,
            "objective": None,
            "declared": {},
        }

    objective = COST_MODE_OBJECTIVE.get(stamped)
    if objective is None:
        raise CostCurrencyError(
            f"{where}: provenance['cost_mode']={stamped!r} names no cost "
            f"objective (known: {sorted(COST_MODE_OBJECTIVE)}); refusing "
            "rather than defaulting")

    mismatched = {
        currency: ROW_CURRENCY_OBJECTIVE[currency]
        for currency in declared
        if ROW_CURRENCY_OBJECTIVE[currency] != objective
    }
    if mismatched:
        raise CostCurrencyError(
            f"{where}: table is stamped COST_MODE={stamped!r} "
            f"(objective {objective!r}) but rows declare "
            + ", ".join(
                f"{c!r} ({declared[c]} rows, objective {o!r})"
                for c, o in sorted(mismatched.items()))
            + "; one DP prices in one currency, and these are not the same "
            "kind of quantity (RobTand/prismaquant#127)")

    return {
        "verdict": VERDICT_SINGLE_CURRENCY,
        "cost_mode": stamped,
        "objective": objective,
        "declared": dict(declared),
    }
