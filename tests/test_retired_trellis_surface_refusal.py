"""A set PRISMAQUANT_TRELLIS_SURFACE must stop the run, not be ignored.

The Gridbook trellis rate surface was archived on 2026-09-02 under
``archive/trellis_wire_2026-09-02/`` (RobTand/prismaquant#118). Deleting the
seam that consumed ``PRISMAQUANT_TRELLIS_SURFACE`` without also refusing on
the variable would leave a driver that still exports it getting a *different*
allocation with no diagnostic at all -- a gate that fails open, which is the
failure class of prismaquant#120. This pins the refusal instead.
"""

from __future__ import annotations

import pytest

from prismaquant import allocator_candidates as ac


ENV = "PRISMAQUANT_TRELLIS_SURFACE"


def test_the_env_name_is_still_spelled_the_way_drivers_spell_it():
    """A renamed constant would silently stop matching stale drivers."""

    assert ac.RETIRED_TRELLIS_SURFACE_ENV == ENV


def test_a_set_surface_flag_refuses_and_names_the_retired_wire(monkeypatch):
    monkeypatch.setenv(ENV, "/home/rob/some-stale-trellis-manifest.json")
    with pytest.raises(ValueError) as excinfo:
        ac.build_candidates({}, {}, [])
    message = str(excinfo.value)
    # The retired wire, by name -- not Tessera's, which is a different
    # plane set and deliberately not a port of it.
    assert "gridbook.trellis.wire.v1" in message
    assert "prismaquant.tessera.v1" in message
    # Where the decision and the bytes went.
    assert "#118" in message
    assert "archive/trellis_wire_2026-09-02/" in message
    # And what the user typed, so a stale driver is findable.
    assert "/home/rob/some-stale-trellis-manifest.json" in message


def test_an_unset_surface_flag_is_not_an_error(monkeypatch):
    """The refusal must fire on the flag, never on an ordinary run."""

    monkeypatch.delenv(ENV, raising=False)
    assert ac.build_candidates({}, {}, []) == {}


def test_an_empty_surface_flag_is_not_an_error(monkeypatch):
    """`export PRISMAQUANT_TRELLIS_SURFACE=` is not a request for the wire."""

    monkeypatch.setenv(ENV, "")
    assert ac.build_candidates({}, {}, []) == {}


def test_the_guard_reads_the_environment_it_is_handed(monkeypatch):
    """Callable without a process env, so the pipeline can pre-flight it."""

    monkeypatch.delenv(ENV, raising=False)
    ac.refuse_retired_trellis_surface({})
    with pytest.raises(ValueError, match=r"#118"):
        ac.refuse_retired_trellis_surface({ENV: "manifest.json"})
