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

Two things this module does not test, on purpose:

* **The dev pin's identity check.**  ``load_tessera_contract`` refuses a
  contract whose bytes are not the ones the pin recorded, and that refusal has
  its own two tests in ``test_tessera_menu.py``.  Here the contract is supplied
  through ``tessera_menu.tessera_runtime_contract`` -- the module's own
  declared "one read" -- built from the bytes the *installed* Tessera actually
  publishes.  That keeps this test about the menu rather than about which
  build is pinned, and it keeps working when the pin is bumped.  (It is also
  the only way it runs today: the dev pin's recorded sha predates Tessera
  ``f6bdb42``, so the pin refuses the installed contract and the attested path
  is dead on this box until the pin moves.)
* **The quality of the allocation.**  The surrogate is measured to mis-rank
  Tessera rungs (``tessera_menu.surrogate_selection_caveat``).  What is
  asserted here is that the DP *saw a menu*, not that it chose well.
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


def _allocate(tmp_path, monkeypatch, *, target_bits="4.0"):
    """Run the allocator's own entry point; return the parsed layer config."""
    from prismaquant import allocator

    layer_config = tmp_path / "layer_config.json"
    monkeypatch.setattr("sys.argv", [
        "allocator",
        "--probe", str(PROBE),
        "--costs", str(COSTS),
        "--formats", "TESSERA",
        "--target-bits", target_bits,
        "--target-profile", "tessera_research_sm121",
        "--layer-config", str(layer_config),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
    ])
    allocator.main()
    return json.loads(layer_config.read_text())


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
