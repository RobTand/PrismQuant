"""A rate-surface anchor set carries a currency, and the DP ranks in one.

RobTand/prismaquant#127: ``tessera_campaign.CURRENCY`` stamps every campaign
row, but nothing downstream read it -- so on the default ``COST_MODE=aura``
path the DP could rank output-MSE-currency Tessera rungs against
AURA-currency NVFP4/FP8/BF16 rungs in one knapsack with no gate saying a
word. This is the trellis seam's refused defect (PR #92,
``trellis_menu._require_run_currency``), inherited by its successor without
the refusal.

The mechanism ported here, not the trellis vocabulary:

* the COST_MODE -> objective-currency table is derived definitionally from
  the COST_RENDER x COST_OBJECTIVE decomposition ``run-pipeline.sh``
  resolves -- never a threshold anyone picks;
* the run's objective is read from the ATTESTED ``provenance['cost_mode']``
  the cost stage stamps -- never from ``os.environ`` (``run-pipeline.sh``
  assigns COST_MODE with ``:=`` and never exports it);
* an unstamped table carrying Tessera rows is refused rather than compared
  against a default, and a COST_MODE naming no objective is refused rather
  than defaulted;
* a Tessera-currency table on a non-render-score run is refused: ranking two
  numbers that are not the same kind of quantity is an error of UNITS, not
  of slope (cf. ``tessera_menu.surrogate_selection_caveat``, which records
  the slope half and must not be conflated with this one).
"""
from __future__ import annotations

import pytest


def _tessera_row():
    from prismaquant.tessera_campaign import CURRENCY

    return {
        "output_mse": 1e-3,
        "output_mse_measured": True,
        "cost_source": "tessera_campaign_measured",
        "currency": CURRENCY,
    }


def _aura_row():
    return {
        "predicted_dloss": 1e-3,
        "weight_mse": 1e-6,
        "cost_currency": "aura_predicted_dloss",
        "cost_source": "production_arm_render",
        "fisher_application_count": 1,
    }


def _payload(cost_mode, rows):
    payload = {"costs": {"u.q_proj": dict(rows)}, "formats": list(rows)}
    if cost_mode is not None:
        payload["provenance"] = {"cost_mode": cost_mode}
    return payload


def test_unstamped_table_with_tessera_rows_refuses():
    """An unstamped cost table is refused rather than compared against a
    default -- the default is exactly what the aura path would assume."""
    from prismaquant.cost_currency import (
        CostCurrencyError,
        require_run_currency,
    )

    with pytest.raises(CostCurrencyError, match="cost_mode"):
        require_run_currency(_payload(None, {"TESSERA_E2M1_K2_R896": _tessera_row()}))


def test_aura_stamped_table_with_tessera_rows_refuses():
    """The issue's acceptance: an aura run ranks in the KL-adjoint, and an
    output-MSE row is not the same kind of quantity."""
    from prismaquant.cost_currency import (
        CostCurrencyError,
        require_run_currency,
    )

    payload = _payload("aura", {
        "NVFP4": _aura_row(),
        "TESSERA_E2M1_K2_R896": _tessera_row(),
    })
    with pytest.raises(CostCurrencyError, match="aura"):
        require_run_currency(payload)


def test_render_score_stamped_table_with_tessera_rows_passes():
    """The campaign measures the render-score objective, so a table stamped
    for it is the one currency the Tessera rows are in."""
    from prismaquant.cost_currency import require_run_currency

    payload = _payload("production-render-score", {
        "NVFP4": {"output_mse": 2e-3, "output_mse_measured": True},
        "TESSERA_E2M1_K2_R896": _tessera_row(),
    })
    report = require_run_currency(payload)
    assert report["expected_currency"] == "render-score"
    assert report["tessera_rows"] == 1


def test_legacy_production_render_alias_passes():
    """`production-render` is the legacy spelling of the same objective
    (`run-pipeline.sh` groups them in one case arm); refusing it would be a
    spelling refusal, not a currency one."""
    from prismaquant.cost_currency import require_run_currency

    payload = _payload("production-render", {
        "TESSERA_E2M1_K2_R896": _tessera_row(),
    })
    assert require_run_currency(payload)["tessera_rows"] == 1


def test_stock_tables_are_outside_this_gate_jurisdiction():
    """No Tessera-currency row, no refusal -- legacy unstamped stock tables
    keep their behavior; other gates own those rows."""
    from prismaquant.cost_currency import require_run_currency

    assert require_run_currency(
        _payload("aura", {"NVFP4": _aura_row()}))["tessera_rows"] == 0
    assert require_run_currency(
        {"costs": {"u": {"NVFP4": _aura_row()}}})["tessera_rows"] == 0


def test_a_cost_mode_naming_no_objective_is_refused():
    from prismaquant.cost_currency import (
        CostCurrencyError,
        require_run_currency,
    )

    with pytest.raises(CostCurrencyError, match="grouped-kl"):
        require_run_currency(_payload(
            "grouped-kl", {"TESSERA_E2M1_K2_R896": _tessera_row()}))


def test_the_expected_currency_comes_from_the_stamp_not_the_environment(monkeypatch):
    """`run-pipeline.sh` assigns COST_MODE with `:=` and never exports it,
    so an environment read compares against a default the run may never have
    used. Both directions: the stamp decides, the environment is ignored."""
    from prismaquant.cost_currency import (
        CostCurrencyError,
        require_run_currency,
    )

    aura_stamp = _payload("aura", {"TESSERA_E2M1_K2_R896": _tessera_row()})
    monkeypatch.setenv("COST_MODE", "production-render-score")
    with pytest.raises(CostCurrencyError, match="aura"):
        require_run_currency(aura_stamp)

    render_stamp = _payload("production-render-score", {
        "TESSERA_E2M1_K2_R896": _tessera_row()})
    monkeypatch.setenv("COST_MODE", "aura")
    assert require_run_currency(render_stamp)["tessera_rows"] == 1


def test_the_tessera_currency_is_read_from_the_campaign_not_restated():
    """Pin the rule, not the roster: the gate reads the currency string from
    the module that stamps it."""
    from prismaquant import cost_currency as cc
    from prismaquant.tessera_campaign import CURRENCY

    assert cc.tessera_campaign_currency() == CURRENCY
    assert CURRENCY == "output_mse_under_route_activation_contract"


def test_allocator_main_refuses_an_aura_table_with_tessera_rows(tmp_path, monkeypatch):
    """Pin the WIRING, not just the helper: deleting the `main()` call site
    must fail this test (cf. `test_allocator_main_enforcement.py`). Drives
    the real `allocator.main()` on a synthetic dense model whose cost table
    is aura-stamped yet carries a Tessera-currency row."""
    import pickle
    import sys

    import prismaquant.allocator as alloc

    names = [f"model.layers.{i}.self_attn.o_proj" for i in range(2)]
    stats = {
        n: {"h_trace": 1.0, "n_params": 4096 * 4096, "shape": [4096, 4096]}
        for n in names
    }
    probe = {"stats": stats, "meta": {"model": None}}
    costs = {
        "costs": {
            n: {
                "NVFP4": dict(_aura_row()),
                "TESSERA_E2M1_K2_R896": dict(_tessera_row()),
            }
            for n in names
        },
        "provenance": {"cost_mode": "aura"},
    }
    probe_p = tmp_path / "probe.pkl"
    cost_p = tmp_path / "cost.pkl"
    probe_p.write_bytes(pickle.dumps(probe))
    cost_p.write_bytes(pickle.dumps(costs))
    monkeypatch.setattr(sys, "argv", [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "NVFP4",
        "--target-bits", "8.0",
        "--pareto-targets", "5.0,8.0",
        "--layer-config", str(tmp_path / "layer_config.json"),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--allow-default-profile",
    ])
    with pytest.raises(SystemExit, match="cost currency"):
        alloc.main()


def test_a_campaign_payload_carries_its_own_cost_mode_stamp():
    """The campaign measures output_mse under the route's activation
    contract -- the render-score objective -- so its payload stamps
    `provenance['cost_mode']` accordingly, and the gate admits its own
    product. An unstamped campaign table would otherwise refuse itself."""
    from prismaquant.cost_currency import require_run_currency
    from prismaquant.tessera_campaign import CampaignAnchor, campaign_cost_payload

    anchor = CampaignAnchor(
        qname="u.q_proj", family="TESSERA_E2M1_K2",
        format_name="TESSERA_E2M1_K2_R896", body_rate_q256=896,
        dloss=1e-4, dloss_stderr=0.0, memory_bytes=1000,
        bits_per_param=3.5, activation_contract="w4a4-nvfp4-e2m1-group16-ue4m3",
        activation_quantized=True, wire_bytes=1000, seconds=1.0,
    )
    payload = campaign_cost_payload(
        {"u.q_proj": {"TESSERA_E2M1_K2": [anchor]}},
        {}, loo={}, provenance={})
    assert payload["provenance"]["cost_mode"] == "production-render-score"
    assert require_run_currency(payload)["tessera_rows"] == 1
