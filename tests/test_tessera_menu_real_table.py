"""The default menu path, end to end, against a cost table a campaign wrote.

Every other Tessera menu test in this tree exercises a *part* of this path
against a list somebody typed, and that is how the 2026-09-02 hole survived:
``expand_menu_tokens`` widened the ``TESSERA`` token to every priced column and
``require_producer_formats`` then refused the **whole** menu, so on the default
(attested) path PrismaQuant could not allocate a Tessera artifact at all.  The
unit tests were green throughout, because each one asked one function about one
hand-written five-element list, and a hand-written list holds the rungs its
author was thinking about -- which is to say the attested ones.  The bug needs
a table priced under ``research`` and read back under ``attested``: two menu
MODES, and no fixture crossed them.  Tessera issue #19.

So this module is deliberately the other kind of test.  It runs the real
allocator entry point over the real 2423-column table the continuous-menu
campaign produced, in the default mode, and asserts the one line the issue
named: **on the default path, with a real table, the attested rung count is
greater than zero** -- plus the line that keeps it from going vacuous, that the
table also holds rungs the runtime does *not* attest.  A table with nothing to
drop cannot reproduce the failure and must not be allowed to look like it did.

The path is walked twice, and the two walks are different tests:

* **Through the one read.**  The contract is supplied through
  ``tessera_menu.tessera_runtime_contract`` -- the module's own declared "one
  read" -- built from the bytes the *installed* Tessera actually publishes.
  That keeps the menu measurement independent of which build the dev pin
  names, so it keeps working when the pin is bumped.  (When this was written
  it was also the only way it ran at all: the pin refused the installed
  contract on its recorded sha, so the attested path was dead on this box
  until someone bumped a constant.  It no longer refuses on bytes -- the pin
  gates on the contract's ANSWER now, issue #38 -- but the one-read supply
  stays, because the reason for it was never the pin's staleness.)
* **Through the dev pin, as an operator runs it.**  ``PRISMAQUANT_TESSERA_
  DEV_PIN`` in the environment, nothing patched.  This is the walk the
  one-read tests are blind to by construction: while they were green, the
  pin's byte-identity check was refusing the installed contract and the
  operator's path was dead (issue #38 -- "PrismaQuant's attested Tessera
  path cannot allocate on this box at all today").  A composed test that
  substitutes the read it is meant to exercise cannot see that, so this one
  substitutes nothing, and asserts on top that the provenance block names the
  table that attested the menu -- which is only true when the allocator reads
  the contract once.

One thing this module does not test, on purpose: **the quality of the
allocation.**  The surrogate is measured to mis-rank Tessera rungs
(``tessera_menu.surrogate_selection_caveat``).  What is asserted here is that
the DP *saw a menu*, not that it chose well.

The campaign predates the required ``provenance.cost_mode`` stamp. Allocation
tests validate its original schema and row currency, then use a temporary
copy with that missing metadata reconstructed from the committed producer.
The source hash and the evidence commits accompany the reconstruction; no
cost value or campaign artifact is changed, and the production gate still
refuses an unstamped table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import format_registry as fr
from prismaquant import tessera_menu as tm
from prismaquant import tessera_runtime_contract as trc

#: The continuous-menu campaign's own artifacts, priced under
#: ``PRISMAQUANT_TESSERA_MENU=research`` on Qwen3-0.6B layer 0 and recorded in
#: ``docs/measurements/tessera-continuous-menu-2026-09-02.md`` §12.  A research
#: table read back on the default path is exactly the cross-mode condition the
#: bug needed, which is why this test wants THIS table and not a synthetic one.
CAMPAIGN = Path("/mnt/shared/tessera-runs/pq-continuous")
PROBE = CAMPAIGN / "qwen06b" / "probe.pkl"
COSTS = CAMPAIGN / "qwen06b_group" / "cost.pkl"

pytestmark = pytest.mark.skipif(
    not (PROBE.exists() and COSTS.exists()),
    reason=(
        f"the campaign's real cost table is not on this box ({PROBE}, {COSTS}); "
        "it lives on /mnt/shared, which both GB10 boxes mount"
    ),
)


@pytest.fixture(autouse=True)
def default_menu_mode(monkeypatch):
    """Run every test here on the DEFAULT menu mode, whatever the shell says.

    The bug this file exists for is a cross-mode one, so a developer with
    ``PRISMAQUANT_TESSERA_MENU=research`` exported would otherwise get a
    harness failure ("not the default path") in place of the measurement.
    """
    monkeypatch.delenv(tm.MENU_MODE_ENV, raising=False)


@pytest.fixture
def installed_contract(monkeypatch):
    """Attest against the contract the installed Tessera actually publishes.

    ``tessera_menu.tessera_runtime_contract`` is documented as the one place
    this module reads a serving contract, so substituting it is substituting
    the whole attestation and nothing else: ``route_admission``,
    ``_producer_eligible``, ``expand_menu_tokens`` and the guard all run
    exactly as production runs them.  ``_load_at`` is the parser
    ``load_tessera_contract`` itself calls, keyed on the file's sha, so the
    contract this returns is the same object the pin would return if the pin
    named this build.
    """
    path = trc.contract_path()
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    contract = trc._load_at(str(path), sha, f"installed:{sha[:12]}")
    monkeypatch.setattr(tm, "tessera_runtime_contract", lambda: contract)
    return contract


def _historical_cost_fixture(tmp_path):
    """Add the producer's later mode stamp to a validated test-only copy.

    ``4d2d9b26`` already requires the scorer to return output_mse and writes
    every row in the currency below. ``7bc4d249`` adds the unconditional
    production-render-score stamp without changing that scoring. These are
    evidence for the reconstruction, not an assertion that either commit
    produced the historical file.
    """
    import pickle

    raw = COSTS.read_bytes()
    original = pickle.loads(raw)
    assert isinstance(original, dict), "historical campaign payload must be a mapping"
    assert original.get("schema") == "prismaquant.tessera_campaign_cost.v1"
    currency = "output_mse_under_route_activation_contract"
    assert original.get("currency") == currency
    provenance = original.get("provenance")
    assert isinstance(provenance, dict), "historical campaign provenance is missing"
    assert "cost_mode" not in provenance, "this fixture is the pre-stamp campaign"
    costs = original.get("costs")
    assert isinstance(costs, dict) and costs, "historical campaign has no costs"
    row_formats = set()
    for unit, rows in costs.items():
        assert isinstance(rows, dict) and rows, f"{unit}: no campaign rows"
        for name, row in rows.items():
            where = f"{unit}/{name}"
            assert isinstance(name, str) and name.startswith("TESSERA_"), where
            assert isinstance(row, dict), where
            assert row.get("currency") == currency, where
            source = row.get("cost_source")
            assert source in {
                "tessera_campaign_measured", "tessera_campaign_interpolated",
            }, where
            measured = source == "tessera_campaign_measured"
            assert row.get("output_mse_measured") is measured, where
            assert row.get("tessera_provenance") == (
                "measured" if measured else "interpolated"), where
            assert "output_mse" in row and "predicted_dloss" not in row, where
            row_formats.add(name)
    formats = original.get("formats")
    assert isinstance(formats, list) and set(formats) == row_formats

    payload = dict(original)
    payload["provenance"] = {
        **provenance,
        "cost_mode": "production-render-score",
        "test_fixture_reconstruction": {
            "source_path": str(COSTS),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "producer_semantics_reviewed_at": "4d2d9b26b64aa9c682724ae701a662a67253ed2f",
            "cost_mode_stamp_introduced_at": "7bc4d249dd7157cccb807f574917b011b0778345",
            "note": (
                "Test-only reconstruction of missing cost_mode from the "
                "original campaign schema, row currencies and committed "
                "producer semantics; all historical cost values are retained."
            ),
        },
    }
    path = tmp_path / "historical-cost-with-mode.pkl"
    path.write_bytes(pickle.dumps(payload))
    assert COSTS.read_bytes() == raw, "historical campaign artifact changed"
    return path


def _allocate(tmp_path, monkeypatch, *, target_bits="4.5"):
    """Run the allocator's own entry point; return the parsed layer config.

    The target sits above the format floor of THIS table.  Over 2423 columns
    the grammar refuses E4M3 R896 (root 7/2 needs 2423/2 columns), so the
    cheapest rung the runtime attests everywhere is R1024 and the allocator
    reports the floor as 4.002 bpp; a 4.0 target is infeasible by that
    0.002, and until prismaquant #126 the accountant never asked the grammar,
    so these tests passed on an allocation Tessera could not build.
    """
    from prismaquant import allocator

    tmp_path.mkdir(parents=True, exist_ok=True)
    costs = _historical_cost_fixture(tmp_path)
    layer_config = tmp_path / "layer_config.json"
    monkeypatch.setattr("sys.argv", [
        "allocator",
        "--probe", str(PROBE),
        "--costs", str(costs),
        "--formats", "TESSERA",
        "--target-bits", target_bits,
        "--target-profile", "tessera_research_sm121",
        "--layer-config", str(layer_config),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
    ])
    allocator.main()
    return json.loads(layer_config.read_text())


def _assignment(config) -> dict[str, str]:
    """``{unit: tessera rung}`` for every unit the DP gave a Tessera rung."""
    return {
        name: entry["tessera_format"]
        for name, entry in config.items()
        if not name.startswith("__") and isinstance(entry, dict)
        and entry.get("tessera_format")
    }


def _installed_contract_sha() -> str:
    """sha256 of the contract file the installed Tessera publishes, read now."""
    return hashlib.sha256(trc.contract_path().read_bytes()).hexdigest()


def test_the_default_path_allocates_over_the_attested_axis(
    tmp_path, monkeypatch, installed_contract,
):
    """The assertion the issue named, on the real table: attested count > 0.

    Regression guard: the fix (PrismaQuant ``0f97ee9``) predates this test, so
    it passes on both sides of that commit.  What it does not pass on is the
    behaviour ``0f97ee9`` removed -- restore the pre-fix expansion (drop the
    ``format_is_producer_eligible`` filter) and this test fails at the guard,
    which is the failure the campaign hit and no unit test did.
    """
    assert not trc.dev_pin_requested(), "this test supplies the contract itself"
    assert tm.menu_mode() == tm.MENU_ATTESTED, "the DEFAULT path, not research"


    config = _allocate(tmp_path, monkeypatch)
    menu = config["__prismaquant__"]["tessera_menu"]

    # The line the issue asked for.
    assert menu["attested_rungs"] > 0, (
        "the default path allocated over an empty Tessera menu: the token "
        "expanded to nothing the pinned runtime attests"
    )
    # ...and the line that stops it being vacuous.  A table holding only
    # attested rungs cannot reproduce the failure, so a future fixture that
    # quietly becomes one must fail here rather than pass silently.
    assert menu["dropped_unattested_rungs"] > 0, (
        f"this table prices {menu['priced_rungs']} rungs and the runtime "
        "attests all of them, so it is not the cross-mode table this test "
        "needs; point it at a research-priced table"
    )
    assert menu["priced_rungs"] > menu["attested_rungs"]

    # The DP saw the menu and used it: a real assignment came out.
    assigned = {
        entry["tessera_format"]
        for name, entry in config.items()
        if not name.startswith("__") and isinstance(entry, dict)
        and entry.get("tessera_format")
    }
    assert assigned, "no unit was assigned a Tessera rung"

    # And every rung it chose survives the guard that refused the whole menu.
    fr.require_producer_formats(sorted(assigned), where="the allocated menu")


def test_the_expansion_reads_the_guards_own_predicate(installed_contract):
    """Lesson 1 of issue #19, pinned behaviourally rather than by reading code.

    The bug's shape was two functions that each looked right and disagreed
    about which set they described.  The repair was to make the token's
    expansion call the predicate the guard refuses on, so this asserts exactly
    that coupling: move the predicate and the expansion moves with it.  A
    second copy of the legality decision -- the thing that caused this -- would
    leave the expansion unchanged here.
    """
    import pickle

    with COSTS.open("rb") as handle:
        priced = [
            name for name in pickle.load(handle)["formats"]
            if isinstance(name, str) and name.startswith("TESSERA_")
        ]
    assert len(priced) > 100, f"expected the campaign's dense table, got {priced[:5]}"

    menu, dropped = tm.expand_menu_tokens_report([tm.MENU_TOKEN], priced)
    assert menu and dropped, (menu[:3], len(dropped))
    assert len(menu) + len(dropped) == len(priced)
    # The expansion is never wider than what the guard accepts.
    fr.require_producer_formats(menu, where="the expanded token")


def test_the_expansion_narrows_with_the_predicate_it_shares(
    monkeypatch, installed_contract,
):
    """Refuse everything at the predicate and the token must expand to nothing.

    This is the coupling test proper: it does not assert *which* rungs are
    attested, only that the token's answer is the guard's answer.
    """
    import pickle

    with COSTS.open("rb") as handle:
        priced = [
            name for name in pickle.load(handle)["formats"]
            if isinstance(name, str) and name.startswith("TESSERA_")
        ]

    monkeypatch.setattr(fr, "format_is_producer_eligible", lambda name: False)
    menu, dropped = tm.expand_menu_tokens_report(["NVFP4", tm.MENU_TOKEN], priced)
    assert menu == ["NVFP4"], menu
    assert len(dropped) == len(priced)


def test_the_provenance_names_the_table_that_attested_the_menu(
    tmp_path, monkeypatch, installed_contract,
):
    """One read: the ``tessera_dev_pin`` block is the contract the menu used.

    ``tessera_menu.tessera_runtime_contract`` is documented as the one place
    a Tessera contract is read, so that "which table answered" is a single
    fact per run.  The allocator's provenance block used to take a second
    read of its own (``load_tessera_contract()`` called directly), which is
    the shape of issue #19 one level up: two functions answering one question
    from two places, agreeing only while nothing distinguishes them.  Supply
    the contract through the one read and the second read answered "no pin",
    so a layer_config whose every Tessera unit was admitted by a contract
    carried no ``tessera_dev_pin`` block at all.  This asserts the block is
    present and names the attesting table's bytes.
    """
    config = _allocate(tmp_path, monkeypatch)
    prov = config["__prismaquant__"]
    assert prov["tessera_menu"]["attested_rungs"] > 0
    assert "tessera_dev_pin" in prov, (
        "units were admitted by a Tessera contract but the provenance says no "
        "contract was read: the allocator took a second read of the pin"
    )
    pin = prov["tessera_dev_pin"]
    assert pin["contract_sha256"] == installed_contract.sha256
    assert pin["commit"] == installed_contract.commit
    assert pin["contract_path"] == installed_contract.path


def test_the_operator_path_allocates_through_the_dev_pin(tmp_path, monkeypatch):
    """The walk an operator takes: the pin in the environment, nothing patched.

    Sets ``PRISMAQUANT_TESSERA_DEV_PIN`` and runs the real entry point over the
    real table, so ``load_tessera_contract`` (the answer check), the admission,
    the token expansion, the guard and the DP all run as production runs them.
    Regression guard for issue #38's route into issue #19's failure: on the
    tree before PrismaQuant ``024588c`` this fails at the pin with
    ``TesseraContractError`` while every one-read test in this file passes.
    """
    monkeypatch.setenv(trc.TESSERA_DEV_PIN_ENV, "1")
    assert trc.dev_pin_requested(), "the pin must be requested for this walk"
    assert tm.menu_mode() == tm.MENU_ATTESTED, "the DEFAULT path, not research"

    config = _allocate(tmp_path, monkeypatch)
    prov = config["__prismaquant__"]
    menu = prov["tessera_menu"]
    assert menu["attested_rungs"] > 0, (
        "the operator's path allocated over an empty Tessera menu"
    )
    assert menu["dropped_unattested_rungs"] > 0, (
        "not a cross-mode table; point this at a research-priced one"
    )

    # The provenance names what the operator asked for and the bytes the run
    # actually read -- the installed contract, whatever the pin's review
    # record says about it (bytes_are_the_reviewed_bytes is a report, not a
    # gate, so it is deliberately not asserted here).
    pin = prov["tessera_dev_pin"]
    assert pin["requested"] == "1"
    assert pin["contract_sha256"] == _installed_contract_sha()
    assert pin["contract_path"] == str(trc.contract_path())

    assigned = _assignment(config)
    assert assigned, "no unit was assigned a Tessera rung"
    fr.require_producer_formats(sorted(set(assigned.values())),
                                where="the operator-path allocation")


def test_the_pin_and_the_one_read_allocate_identically(tmp_path, monkeypatch):
    """Same contract by two routes, same allocation.

    The pin gates on the contract's answer and the one-read fixture hands the
    allocator the same contract object the pin would return, so the two walks
    must produce the same menu counts and the same per-unit assignment.  If
    they ever differ, one of the two routes is reading something the other
    is not -- a second copy of the attestation -- and this says so before a
    shipcard does.
    """
    monkeypatch.setenv(trc.TESSERA_DEV_PIN_ENV, "1")
    via_pin = _allocate(tmp_path / "pin", monkeypatch)

    monkeypatch.delenv(trc.TESSERA_DEV_PIN_ENV)
    path = trc.contract_path()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = trc._load_at(str(path), sha, f"installed:{sha[:12]}")
    monkeypatch.setattr(tm, "tessera_runtime_contract", lambda: contract)
    via_read = _allocate(tmp_path / "read", monkeypatch)

    counts = ("priced_rungs", "attested_rungs", "dropped_unattested_rungs")
    pin_menu = via_pin["__prismaquant__"]["tessera_menu"]
    read_menu = via_read["__prismaquant__"]["tessera_menu"]
    assert {k: pin_menu[k] for k in counts} == {k: read_menu[k] for k in counts}
    assert pin_menu["attested_rungs"] > 0
    assert _assignment(via_pin) == _assignment(via_read)
    assert _assignment(via_pin)
    assert (via_pin["__prismaquant__"]["tessera_dev_pin"]["contract_sha256"]
            == via_read["__prismaquant__"]["tessera_dev_pin"]["contract_sha256"]
            == sha)
