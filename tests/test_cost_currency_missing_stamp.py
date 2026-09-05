"""A Tessera format cannot hide from its currency gate by dropping a stamp."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("currency", [None, "", "not-the-campaign-currency"])
@pytest.mark.parametrize("mode", ["production-render-score", "aura"])
def test_tessera_format_requires_its_owned_currency_stamp(currency, mode):
    from prismaquant.cost_currency import CostCurrencyError, require_run_currency
    row = {"output_mse": 1e-4, "output_mse_measured": True}
    if currency is not None:
        row["currency"] = currency
    payload = {
        "provenance": {"cost_mode": mode},
        "costs": {"named.unit": {"TESSERA_E4M3_K1_R1024": row}},
    }
    with pytest.raises(CostCurrencyError, match=r"named\.unit.*currency|currency.*named\.unit"):
        require_run_currency(payload)


def test_error_row_is_not_a_price_requiring_currency():
    from prismaquant.cost_currency import require_run_currency
    payload = {"costs": {"unit": {
        "TESSERA_E4M3_K1_R1024": {"error": "not measured"},
        "BF16": {"predicted_dloss": 0.0},
    }}}
    assert require_run_currency(payload)["tessera_rows"] == 0


def test_stock_row_is_not_reclassified_as_tessera_by_a_prefix_guess():
    from prismaquant.cost_currency import require_run_currency
    # The actual grammar, not startswith("TESSERA"), owns membership.
    payload = {"costs": {"unit": {"TESSERA_unrelated": {"output_mse": 1e-4}}}}
    assert require_run_currency(payload)["tessera_rows"] == 0
