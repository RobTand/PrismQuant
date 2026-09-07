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

#: The allocation walks need the campaign's artifacts; the contract reads do
#: not.  This was a module-level ``pytestmark`` until the readable-menu work
#: (#284) added tests that ask only the pinned contract what its decoder
#: accepts.  Skipping those for want of a 7 MB probe pickle would have made
#: the one assertion the issue is about read "did not run" on any box that
#: does not mount ``/mnt/shared`` -- a green receipt for a test nothing ran.
campaign_table = pytest.mark.skipif(
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


def _default_serve_image() -> str:
    """The serve image the installed contract publishes as its default."""
    return json.loads(trc.contract_path().read_text(encoding="utf-8"))[
        "versions"]["default_serve_image"]


def _scope_flags() -> list[str]:
    """The operator's serving scope, as ``run-pipeline.sh`` passes it.

    The pinned table is scoped (lane schema v8): a cell attests a rung only
    at an exact platform, image, execution mode and residency, so the
    allocator is told which serve this allocation is for.  Dense, resident,
    eager, on the contract's own default serve image -- the scope the eight
    dense cells publish.  Without these flags the token expands to nothing
    and the allocator refuses (``tests/test_tessera_menu.py`` pins that).
    """
    return [
        "--tessera-platform", "sm_121",
        "--tessera-runtime-image", _default_serve_image(),
        "--tessera-execution-mode", "eager",
        "--tessera-residency", "resident",
    ]


def _dense_context():
    from prismaquant.lane_eligibility import ServingContext
    return ServingContext(
        platform="sm_121", structure="dense", residency="resident",
        runtime_image=_default_serve_image(), execution_mode="eager")


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
        *_scope_flags(),
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


@campaign_table
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

    # And every rung it chose survives the guard that refused the whole menu,
    # asked at the scope the allocation was made for.
    fr.require_producer_formats(
        sorted(assigned), where="the allocated menu",
        context_by_unit={unit: _dense_context() for unit in assigned})


@campaign_table
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

    # The scoped table answers only at a scope; the token and the guard are
    # asked the same one, and context-free both refuse everything (the
    # expansion narrows to nothing, which the allocator reports by name).
    scope = {"unit": _dense_context()}
    menu, dropped = tm.expand_menu_tokens_report([tm.MENU_TOKEN], priced, context_by_unit=scope)
    assert menu and dropped, (menu[:3], len(dropped))
    assert len(menu) + len(dropped) == len(priced)
    # The expansion is never wider than what the guard accepts.
    fr.require_producer_formats(menu, where="the expanded token", context_by_unit=scope)
    unscoped, all_dropped = tm.expand_menu_tokens_report([tm.MENU_TOKEN], priced)
    assert unscoped == [] and len(all_dropped) == len(priced)


@campaign_table
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

    monkeypatch.setattr(fr, "format_is_producer_eligible", lambda name, **scope: False)
    menu, dropped = tm.expand_menu_tokens_report(
        ["NVFP4", tm.MENU_TOKEN], priced, context_by_unit={"unit": _dense_context()})
    assert menu == ["NVFP4"], menu
    assert len(dropped) == len(priced)


@campaign_table
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


@campaign_table
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
                                where="the operator-path allocation",
                                context_by_unit={unit: _dense_context() for unit in assigned})


@campaign_table
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


# ---------------------------------------------------------------------------
# The readable menu, on the contract the installed Tessera actually publishes
# ---------------------------------------------------------------------------

#: What contract v22's ``formats[].reader_rate_range_q256`` says the plugin's
#: decoder accepts, read here as the EXPECTATION the two code paths must both
#: reproduce. It is written out rather than re-read from the file on purpose:
#: a test that derives its expectation from the same field the code reads
#: asserts only that one read happened twice.
READER_RANGES = {
    "TESSERA_E2M1_K2": (896, 896),
    "TESSERA_E4M3_K1": (256, 2048),
    "TESSERA_BF16_K1": (256, 4096),
}
#: And the family the contract does not publish at all -- the one the #275 LFM
#: campaign spent 168 rows x 27 s encoding. Absence is "no reader exists",
#: never "any rate is fine", so every rung of it must be refused.
UNPUBLISHED_FAMILY = "TESSERA_E2M1_K1"


def _readable_by_family(*, families=None):
    """``{family: {rate: readable}}`` over every legal rung of every family.

    Walks the RESEARCH menu, which is the whole shape-legal set, so the
    "exactly" in the acceptance criterion is a statement about every rung the
    producer can write and not about a list somebody typed.
    """
    out: dict[str, dict[int, bool]] = {}
    for rung in tm.expand_tessera_menu(
        (2048, 1024), mode=tm.MENU_RESEARCH,
        **({"families": families} if families is not None else {}),
    ):
        out.setdefault(rung.family, {})[rung.body_rate_q256] = (
            rung.admission.readable)
    return out


def _assert_the_published_reader_ranges(seen):
    assert set(seen) >= set(READER_RANGES) | {UNPUBLISHED_FAMILY}, sorted(seen)
    for family, rates in sorted(seen.items()):
        span = READER_RANGES.get(family)
        readable = {rate for rate, ok in rates.items() if ok}
        if span is None:
            assert not readable, (
                f"{family} is not published by the pinned contract, so no "
                f"reader accepts it; {sorted(readable)[:8]} were admitted")
            continue
        lo, hi = span
        expected = {rate for rate in rates if lo <= rate <= hi}
        assert readable == expected, (
            f"{family}: readable={sorted(readable)[:8]} "
            f"expected={sorted(expected)[:8]} over range {span}")
        assert expected, f"{family}: the shape carries no rung inside {span}"


def test_the_readable_menu_is_exactly_the_published_reader_ranges(
    installed_contract,
):
    """The set the issue named, through the module's declared one read.

    ``TESSERA_E2M1_K2`` at 896 only; ``TESSERA_E4M3_K1`` over [256, 2048];
    ``TESSERA_BF16_K1`` over [256, 4096]; and nothing at all from
    ``TESSERA_E2M1_K1``, which the contract does not publish.
    """
    seen = _readable_by_family()
    # Not vacuous: the family the campaign wasted its time on IS on the
    # research menu, at many rungs, and every one of them must be refused.
    assert len(seen.get(UNPUBLISHED_FAMILY, {})) > 50, (
        "the unpublished family is absent from the research menu, so "
        "'refuses every E2M1_K1 rung' asserts nothing")
    _assert_the_published_reader_ranges(seen)
    # The two anchors the #275 campaign encoded outside the reader range.
    for rate in (128, 512):
        assert seen["TESSERA_E2M1_K2"][rate] is False
    assert seen["TESSERA_E2M1_K2"][896] is True


def test_the_packaged_branch_agrees_with_the_dev_pin_on_readability():
    """The same answer with no pin supplied: the packaged ``formats`` table.

    ``route_admission`` derives ``readable`` from two different fields on two
    different objects -- ``reader_rate_range`` on the parsed dev-pin contract,
    ``reader_rate_range_q256`` on the packaged ``formats`` row. Nothing else
    in this suite crosses those two paths, so a wrong key on either one is
    invisible until a campaign prices the wrong axis. This runs the walk with
    no contract substituted and no pin requested, which is the branch
    production takes, and holds it to the same expectation.
    """
    assert not trc.dev_pin_requested(), "this walk takes the packaged branch"
    _assert_the_published_reader_ranges(_readable_by_family())


def test_attested_is_contained_in_readable_is_contained_in_research(
    installed_contract,
):
    """``attested <= readable <= research``, rung by rung, family by family.

    Asserted as containment of the admission predicate rather than of three
    menus, so a family whose shape drops every rung cannot make the ordering
    hold by being empty on all three sides.
    """
    context = _dense_context()
    # The scoped table attests a cell only under an exact serving scope, so a
    # context-free walk would count zero attested rungs and the containment
    # would hold by being empty on one side.
    attested_seen = readable_seen = 0
    for rung in tm.expand_tessera_menu(
        (2048, 1024), mode=tm.MENU_RESEARCH, serving_context=context,
    ):
        admission = rung.admission
        assert admission.admits(tm.MENU_RESEARCH), rung.format_name
        if admission.admits(tm.MENU_ATTESTED):
            attested_seen += 1
            assert admission.admits(tm.MENU_READABLE), (
                f"{rung.format_name} is attested by a cell the contract "
                "publishes and yet its own decoder is said to refuse the rate")
        if admission.admits(tm.MENU_READABLE):
            readable_seen += 1
        # The containment that matters for the export gate: readable moves
        # nothing.  A rung nothing attests is `unattested` in every mode.
        if not admission.attested:
            assert admission.route_status == tm.ROUTE_STATUS_UNATTESTED
    assert attested_seen > 0 and readable_seen > attested_seen, (
        attested_seen, readable_seen)


def test_readability_does_not_depend_on_the_serving_scope(installed_contract):
    """A decoder either takes a rate or it does not; a scope cannot change it.

    ``attested`` is scope-bound by construction -- a cell attests one platform,
    image, execution mode and residency -- and ``readable`` deliberately is
    not, because ``reader_rate_range_q256`` is published once per family and
    says nothing about where the rung is served. Asserting it here keeps the
    two from quietly merging: a ``readable`` that moved with the scope would be
    a weaker spelling of ``attested`` wearing a different name.
    """
    context = _dense_context()
    for name in ("TESSERA_E4M3_K1_R512", "TESSERA_E4M3_K1_R1024",
                 "TESSERA_E2M1_K2_R512", "TESSERA_E2M1_K2_R896"):
        scoped = tm.route_admission(name, serving_context=context)
        bare = tm.route_admission(name)
        assert scoped.readable == bare.readable, name
    # ...and the attestation is, which is what makes the above non-trivial.
    assert tm.route_admission(
        "TESSERA_E4M3_K1_R1024", serving_context=context).attested
    assert not tm.route_admission("TESSERA_E4M3_K1_R1024").attested
