from __future__ import annotations

import json

import pytest

import prismaquant.shipcard as shipcard


def _git_unavailable(*_args, **_kwargs):
    raise FileNotFoundError("fixture has no git metadata")


def test_export_shipcard_uses_explicit_clean_git_identity_without_worktree(
    tmp_path, monkeypatch,
):
    """Renamed and re-pointed on 2026-09-02.

    This drove `shipcard.open_cb_export_shipcard`, the Gridbook lane's
    card-opening wrapper, which retired with the lane
    (archive/gridbook_lane_2026-09-02/). The subject is the *identity*, not the
    wrapper: an exporter running where `git` is unavailable must still stamp a
    clean commit from `PRISMAQUANT_IDENTITY_GIT_*` rather than inventing one.
    So it now composes the card exactly the way the surviving native exporter
    does -- `export_native_compressed.py:8494` sets `build["git"] =
    git_provenance()` and hands it to `build_shipcard` -- which is the live
    path this assertion is about.
    """
    model = tmp_path / "artifact"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture-weight")
    monkeypatch.setattr(shipcard.subprocess, "run", _git_unavailable)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_DIRTY", "0")

    card = shipcard.build_shipcard(
        model, build={"git": shipcard.git_provenance()}
    )
    path = shipcard.write_shipcard(tmp_path / "shipcard.json", card)
    assert shipcard.load_shipcard(path)["build"]["git"] == {
        "commit": "a" * 40,
        "dirty": False,
    }


def test_git_override_without_clean_preflight_does_not_invent_dirty_false(
    monkeypatch,
):
    monkeypatch.setattr(shipcard.subprocess, "run", _git_unavailable)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "b" * 40)
    monkeypatch.delenv("PRISMAQUANT_IDENTITY_GIT_DIRTY", raising=False)
    assert shipcard.git_provenance() == {
        "commit": "b" * 40,
        "dirty": None,
    }


def test_git_overrides_refuse_contradictory_mounted_worktree(
    monkeypatch,
):
    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(command, **_kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return _Result("c" * 40 + "\n")
        if command[1:3] == ["status", "--short"]:
            return _Result(" M prismaquant/shipcard.py\n")
        raise AssertionError(command)

    monkeypatch.setattr(shipcard.subprocess, "run", fake_run)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "c" * 40)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_DIRTY", "0")
    with pytest.raises(ValueError, match="contradicts.*dirty=True"):
        shipcard.git_provenance()

