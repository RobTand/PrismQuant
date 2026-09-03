"""One DP prices in one currency, and the currency is a value a gate reads.

RobTand/prismaquant#127: ``tessera_campaign`` stamped
``currency = output_mse_under_route_activation_contract`` on every row and
nothing read it, so under the default ``COST_MODE=aura`` a DP could rank
Fisher-transferred output MSE against KL-adjoint projections in one
``max()``.  These tests pin the RULE -- declared currency must name the
attested objective -- and never a roster: the mode table is held equal to
``run-pipeline.sh``'s own ``case`` block, and every producer's spelling is
held equal to the leaf module's.
"""
from __future__ import annotations

import pathlib
import pickle
import re
import sys

import pytest

from prismaquant import cost_currency as cc

ROOT = pathlib.Path(__file__).resolve().parents[1]

TESSERA_ROW = {
    "output_mse": 1e-4, "output_mse_measured": True,
    "cost_source": "tessera_campaign_measured",
    "currency": cc.TESSERA_CAMPAIGN_CURRENCY,
}
STOCK_ROW = {"weight_mse": 1e-4, "output_mse": 1e-4,
             "output_mse_measured": True, "predicted_dloss": 1e-4}
ANCHORED_ROW = {"predicted_dloss": 1e-4,
                "cost_currency": cc.ANCHORED_AURA_COST_CURRENCY,
                "cost_source": "production_arm_render",
                "fisher_application_count": 1}


def _table(rows: dict, cost_mode: str | None = None, **extra) -> dict:
    payload = {"costs": {"m.0.q_proj": rows}, **extra}
    if cost_mode is not None:
        payload["provenance"] = {"cost_mode": cost_mode}
    return payload


# ---------------------------------------------------------------------------
# The two tables are derived, not restated
# ---------------------------------------------------------------------------

def test_the_mode_table_is_the_pipelines_own_case_block():
    """``COST_MODE_OBJECTIVE`` mirrors ``run-pipeline.sh``; the shell is the authority.

    Parsed from the script rather than retyped, so a mode added or renamed in
    one spelling and not the other fails here instead of in a DP.
    """
    src = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    block = src[src.index('case "$COST_MODE" in'):]
    block = block[:block.index("esac")]
    parsed: dict[str, str] = {}
    for match in re.finditer(
            r"^\s*([a-z][a-z0-9|-]*)\)\s*\n?\s*COST_RENDER=\S+;\s*COST_OBJECTIVE=(\S+)",
            block, re.MULTILINE):
        for mode in match.group(1).split("|"):
            parsed[mode] = match.group(2)
    assert parsed, "could not parse the COST_MODE case block"
    assert "*" not in parsed
    assert parsed == dict(cc.COST_MODE_OBJECTIVE)


def test_every_producer_spells_its_currency_from_the_leaf():
    """The writer and the gate read one string."""
    from prismaquant import allocator_candidates as ac
    from prismaquant import anchored_cost
    from prismaquant import tessera_campaign as tc

    assert tc.CURRENCY is cc.TESSERA_CAMPAIGN_CURRENCY
    assert ac.ANCHORED_AURA_COST_CURRENCY is cc.ANCHORED_AURA_COST_CURRENCY
    assert anchored_cost.AURA_CURRENCY is cc.ANCHORED_AURA_COST_CURRENCY
    # and every declared currency names an objective some COST_MODE prices in
    for currency, objective in cc.ROW_CURRENCY_OBJECTIVE.items():
        assert objective in set(cc.COST_MODE_OBJECTIVE.values()), currency
        assert cc.COST_MODE_OBJECTIVE[cc.cost_mode_for_currency(currency)] == objective


def test_the_campaign_derives_its_stamp_from_its_currency():
    from prismaquant import tessera_campaign as tc

    assert tc.COST_MODE == cc.cost_mode_for_currency(tc.CURRENCY)
    assert cc.COST_MODE_OBJECTIVE[tc.COST_MODE] == tc.COST_OBJECTIVE
    assert tc.COST_OBJECTIVE != cc.COST_MODE_OBJECTIVE["aura"], (
        "the campaign's own header says it is not the AURA adjoint")


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def test_an_aura_table_carrying_tessera_rows_refuses():
    """The finding itself: output MSE and a KL-adjoint in one max()."""
    table = _table({"TESSERA_E2M1_K2_R896": dict(TESSERA_ROW),
                    "NVFP4": dict(STOCK_ROW)}, cost_mode="aura")
    with pytest.raises(cc.CostCurrencyError, match="one DP prices in one currency"):
        cc.require_cost_currency(table)


def test_an_unstamped_table_carrying_a_declared_currency_refuses():
    """Refused, not compared against a default."""
    table = _table({"TESSERA_E2M1_K2_R896": dict(TESSERA_ROW)})
    with pytest.raises(cc.CostCurrencyError, match="no provenance\\['cost_mode'\\]"):
        cc.require_cost_currency(table)
    # the table-level stamp alone is enough to require a mode
    with pytest.raises(cc.CostCurrencyError, match="no provenance"):
        cc.require_cost_currency(
            _table({"NVFP4": dict(STOCK_ROW)}, currency=cc.TESSERA_CAMPAIGN_CURRENCY))


def test_a_mode_naming_no_objective_refuses_rather_than_defaulting():
    for mode in ("grouped-kl", "", "AURA", "production_render_score"):
        table = _table({"NVFP4": dict(STOCK_ROW)}, cost_mode=mode or None)
        if not mode:
            # empty stamp == unstamped; with no declared rows that is legacy
            assert cc.require_cost_currency(table)["verdict"] == cc.VERDICT_UNDECLARED_UNSTAMPED
            continue
        with pytest.raises(cc.CostCurrencyError, match="names no cost objective"):
            cc.require_cost_currency(table)


def test_an_unknown_row_currency_refuses():
    row = {**TESSERA_ROW, "currency": "furlongs_per_fortnight"}
    with pytest.raises(cc.CostCurrencyError, match="unknown unit"):
        cc.require_cost_currency(_table({"X": row}, cost_mode="aura"))


def test_a_row_declaring_two_disagreeing_currencies_refuses():
    row = {**TESSERA_ROW, "cost_currency": cc.ANCHORED_AURA_COST_CURRENCY}
    with pytest.raises(cc.CostCurrencyError, match="two currencies"):
        cc.require_cost_currency(_table({"X": row}, cost_mode="aura"))


def test_the_gate_is_symmetric_across_currencies():
    """An anchored-AURA row on a render-score table is the same defect mirrored."""
    table = _table({"NVFP4": dict(ANCHORED_ROW)}, cost_mode="production-render-score")
    with pytest.raises(cc.CostCurrencyError, match="one DP prices in one currency"):
        cc.require_cost_currency(table)
    ok = cc.require_cost_currency(_table({"NVFP4": dict(ANCHORED_ROW)}, cost_mode="aura"))
    assert ok["verdict"] == cc.VERDICT_SINGLE_CURRENCY
    assert ok["objective"] == "aura-adjoint"


# ---------------------------------------------------------------------------
# What passes, and what it records
# ---------------------------------------------------------------------------

def test_a_table_stamped_in_the_rows_own_objective_passes_and_records_it():
    table = _table({"TESSERA_E2M1_K2_R896": dict(TESSERA_ROW),
                    "NVFP4": dict(STOCK_ROW)},
                   cost_mode="production-render-score",
                   currency=cc.TESSERA_CAMPAIGN_CURRENCY)
    verdict = cc.require_cost_currency(table)
    assert verdict == {
        "verdict": cc.VERDICT_SINGLE_CURRENCY,
        "cost_mode": "production-render-score",
        "objective": "render-score",
        "declared": {cc.TESSERA_CAMPAIGN_CURRENCY: 1},
    }
    # the alias spelling is the same objective
    table["provenance"]["cost_mode"] = "production-render"
    assert cc.require_cost_currency(table)["objective"] == "render-score"


def test_a_legacy_table_with_nothing_to_compare_passes_and_says_so():
    verdict = cc.require_cost_currency(_table({"NVFP4": dict(STOCK_ROW)}))
    assert verdict["verdict"] == cc.VERDICT_UNDECLARED_UNSTAMPED
    assert verdict["cost_mode"] is None and verdict["declared"] == {}


def test_the_expected_currency_comes_from_the_table_not_the_environment(monkeypatch):
    """``COST_MODE`` in the environment must not move the verdict either way."""
    aura_table = _table({"TESSERA_E2M1_K2_R896": dict(TESSERA_ROW)}, cost_mode="aura")
    good_table = _table({"TESSERA_E2M1_K2_R896": dict(TESSERA_ROW)},
                        cost_mode="production-render-score")
    for env in (None, "aura", "production-render-score", "local", "nonsense"):
        if env is None:
            monkeypatch.delenv("COST_MODE", raising=False)
        else:
            monkeypatch.setenv("COST_MODE", env)
        with pytest.raises(cc.CostCurrencyError):
            cc.require_cost_currency(aura_table)
        assert cc.require_cost_currency(good_table)["objective"] == "render-score"


def test_the_campaign_payload_is_stamped_and_passes_the_gate():
    """The producer half: every campaign table carries the attested mode."""
    from prismaquant.tessera_campaign import CampaignAnchor, campaign_cost_payload

    q, fam = "m.0.q_proj", "TESSERA_E2M1_K2"
    anchors = [
        CampaignAnchor(qname=q, family=fam, body_rate_q256=r,
                       format_name=f"{fam}_R{r}", dloss=d, dloss_stderr=d / 10,
                       activation_quantized=True, activation_contract="W4A4",
                       wire_bytes=1, seconds=0.0, hessian_applied=False)
        for r, d in ((128, 1e-2), (512, 1e-3), (896, 1e-4))
    ]
    payload = campaign_cost_payload({q: {fam: anchors}}, {}, loo={}, provenance={})
    assert payload["provenance"]["cost_mode"] == "production-render-score"
    assert payload["provenance"]["cost_objective"] == "render-score"
    assert cc.require_cost_currency(payload)["verdict"] == cc.VERDICT_SINGLE_CURRENCY
    # a caller that pre-stamped a different mode is refused, not overwritten
    with pytest.raises(cc.CostCurrencyError, match="refusing to stamp"):
        campaign_cost_payload({q: {fam: anchors}}, {}, loo={},
                              provenance={"provenance": {"cost_mode": "aura"}})
    # and pre-stamping the right one is idempotent
    same = campaign_cost_payload(
        {q: {fam: anchors}}, {}, loo={},
        provenance={"provenance": {"cost_mode": "production-render-score"}})
    assert same["provenance"]["cost_mode"] == "production-render-score"


# ---------------------------------------------------------------------------
# The gate is wired into allocator.main(), before any candidate is built
# ---------------------------------------------------------------------------

def _allocator_fixture(tmp_path, cost_mode):
    names = [f"model.layers.{i}.self_attn.o_proj" for i in range(2)]
    stats = {n: {"h_trace": 1.0 + 0.1 * i, "n_params": 4096 * 4096,
                 "shape": [4096, 4096]} for i, n in enumerate(names)}
    probe = {"stats": stats, "meta": {"model": None}}
    costs = {
        "costs": {n: {"NVFP4": dict(STOCK_ROW), "FP8_E4M3": dict(STOCK_ROW),
                      "TESSERA_E2M1_K2_R896": dict(TESSERA_ROW)}
                  for n in names},
        "meta": {"formats": ["NVFP4", "FP8_E4M3"]},
        "provenance": {"cost_mode": cost_mode},
    }
    probe_p, cost_p = tmp_path / "probe.pkl", tmp_path / "cost.pkl"
    probe_p.write_bytes(pickle.dumps(probe))
    cost_p.write_bytes(pickle.dumps(costs))
    return probe_p, cost_p


def test_allocator_main_refuses_before_building_candidates(tmp_path, monkeypatch):
    import prismaquant.allocator as alloc

    probe_p, cost_p = _allocator_fixture(tmp_path, "aura")
    built = []
    monkeypatch.setattr(alloc, "build_candidates",
                        lambda *a, **k: built.append(1) or {})
    monkeypatch.setattr(sys, "argv", [
        "allocator", "--probe", str(probe_p), "--costs", str(cost_p),
        "--target-bits", "5.0", "--formats", "NVFP4,FP8_E4M3",
        "--layer-config", str(tmp_path / "lc.json"),
        "--pareto-csv", str(tmp_path / "p.csv"),
    ])
    with pytest.raises(SystemExit) as exc:
        alloc.main()
    assert "one DP prices in one currency" in str(exc.value)
    assert built == [], "the DP saw candidates from a table the gate refused"
