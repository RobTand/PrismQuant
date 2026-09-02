from __future__ import annotations

import json

import numpy as np
import pytest
import tools.dsv4_afast_campaign as afast_campaign

from tools.dsv4_afast_campaign import (
    RSSGuard,
    RSSLimitExceeded,
    _cell_identity,
    _quarantine_projection_suffix,
    _sha256_text,
    _validated_cell_shard,
    _unit_boundary_reclaim,
    pchip_monotone,
)
from tools.dsv4_ldlq_cost_campaign import atomic_pickle
from tools.dsv4_afast_burn import (
    ANCHORS,
    BACKSTOP_TOLERANCE,
    BURN_CELL_IDENTITY_SCHEMA,
    BURN_CELL_SCHEMA,
    BURN_PASS_TAG_SCHEMA,
    BURN_PASS_TAGS,
    EXPERT_COUNT,
    _acceptance,
    _audit_rung,
    _burn_cell_identity,
    _burn_cell_path,
    _sha,
    _validated_burn_cell,
    _validated_burn_cell_or_invalidate,
)
from tools.dsv4_afast_allocation_grid import (
    _complete as complete_grid_cell,
    _validated_completion,
)


def test_pchip_preserves_anchors_and_monotonicity():
    anchors = (28, 33, 38, 43, 48)
    values = (10.0, 8.0, 7.0, 4.0, 3.0)
    query = tuple(range(28, 49))
    prediction = pchip_monotone(anchors, values, query)
    assert np.all(np.diff(prediction) <= 1e-12)
    for rung, value in zip(anchors, values):
        assert prediction[rung - 28] == value


def test_pchip_isotonicizes_epsilon_jittered_anchors():
    prediction = pchip_monotone(
        (28, 33, 38, 43, 48),
        (10.0, 8.0, 8.0 + 1e-13, 4.0, 3.0),
        tuple(range(28, 49)),
    )
    assert np.all(np.diff(prediction) <= 1e-12)


def test_afast_backstop_accepts_smooth_curve_and_rejects_gross_outlier():
    # A smooth exponential decay is *not* law-exact: the per-expert level-2
    # fit absorbs its slope but not the level-1 phi staircase, so this
    # exercises the backstop's headroom over ordinary law residue rather
    # than over a synthetic perfect fit.  Rungs come from the live anchor
    # set plus a real audit draw, so a domain move carries through.
    audit_rung = _audit_rung(0)
    assert audit_rung not in ANCHORS

    def curve(rung: int) -> float:
        return 3.0 * 2.0 ** (-rung / 5.0)

    anchor_errors = {rung: [curve(rung)] * EXPERT_COUNT for rung in ANCHORS}
    audit_errors = [curve(audit_rung)] * EXPERT_COUNT

    accepted, rejected, report = _acceptance(
        anchor_errors, audit_rung, audit_errors, "gate_proj",
    )
    assert len(accepted) == EXPERT_COUNT
    assert not rejected
    assert report["accepted"] == EXPERT_COUNT
    assert report["rejected"] == 0
    assert max(
        row["audit_rung_relative_error"] for row in report["per_slice"]
    ) <= BACKSTOP_TOLERANCE

    # Miss the audit rung by a margin midway between the bar and a total
    # miss -- gross under any tolerance the operator may register.
    gross = (1.0 + BACKSTOP_TOLERANCE) / 2.0
    audit_errors = list(audit_errors)
    audit_errors[17] /= 1.0 - gross

    accepted, rejected, report = _acceptance(
        anchor_errors, audit_rung, audit_errors, "gate_proj",
    )
    assert rejected == [17]
    assert 17 not in accepted
    assert report["rejected"] == 1
    assert report["accepted"] == EXPERT_COUNT - 1
    assert sorted(accepted + rejected) == list(range(EXPERT_COUNT))
    outlier = next(row for row in report["per_slice"] if row["expert"] == 17)
    assert outlier["audit_rung_relative_error"] > BACKSTOP_TOLERANCE


def test_cell_shard_resume_is_content_keyed(tmp_path):
    warm = tmp_path / "warm.safetensors"
    warm.write_bytes(b"warm")
    identity = _cell_identity(
        layer=14,
        projection="gate_proj",
        rung=29,
        expert_ids=(0, 1),
        content_guard={"source_digest": "a" * 64},
        predecessor_content_key="b" * 64,
    )
    path = tmp_path / "cell.pkl"
    atomic_pickle(path, {
        "schema": "prismaquant.dsv4_afast_cell_shard.v1",
        "content_key": _sha256_text(identity),
        "identity": identity,
        "cell": {
            "rung": 29,
            "expert_ids": [0, 1],
            "warm_state_path": str(warm),
        },
    })
    assert _validated_cell_shard(path, identity) is not None
    stale = dict(identity, predecessor_content_key="c" * 64)
    with pytest.raises(AssertionError, match="stale or corrupt"):
        _validated_cell_shard(path, stale)


def test_rss_guard_persists_first_crossing_evidence(tmp_path):
    path = tmp_path / "rss.json"
    guard = RSSGuard(path, limit_bytes=1)
    guard.stage = "unit-test"
    with pytest.raises(RSSLimitExceeded):
        guard.checkpoint()
    evidence = json.loads(path.read_text())
    assert evidence["stage"] == "unit-test"
    assert evidence["rss_bytes"] > evidence["limit_bytes"]


def test_unit_boundary_releases_cuda_cache_only_below_host_threshold(monkeypatch):
    releases = []
    monkeypatch.setattr(afast_campaign, "_malloc_trim", lambda: None)
    monkeypatch.setattr(
        afast_campaign.torch.cuda, "empty_cache", lambda: releases.append(True),
    )
    monkeypatch.setattr(
        afast_campaign, "_host_available_bytes", lambda: 41 * 1024**3,
    )
    high = _unit_boundary_reclaim(unit="L14:gate_proj:K42")
    assert high["cuda_cache_released"] is False
    assert releases == []

    monkeypatch.setattr(
        afast_campaign, "_host_available_bytes", lambda: 39 * 1024**3,
    )
    low = _unit_boundary_reclaim(unit="L14:gate_proj:K43")
    assert low["cuda_cache_released"] is True
    assert releases == [True]


def test_burn_cell_resume_binds_phase_encoded_set_and_predecessor(tmp_path):
    warm = tmp_path / "warm.safetensors"
    warm.write_bytes(b"warm")
    identity = _burn_cell_identity(
        layer=0, projection="down_proj",
        pass_tag=BURN_PASS_TAGS["backstop"], rung=37,
        expert_ids=(3, 9), encoded_expert_ids=(3, 9),
        content_guard={"source_digest": "a" * 64},
        predecessor_content_key="b" * 64, replay=True,
    )
    path = tmp_path / "burn.pkl"
    atomic_pickle(path, {
        "schema": BURN_CELL_SCHEMA,
        "pass_tag_schema": BURN_PASS_TAG_SCHEMA,
        "pass_tag": BURN_PASS_TAGS["backstop"],
        "content_key": _sha(identity), "identity": identity,
        "cell": {
            "rung": 37, "expert_ids": [3, 9],
            "warm_state_path": str(warm),
        },
    })
    assert _validated_burn_cell(path, identity) is not None
    for stale in (
        dict(identity, pass_tag=BURN_PASS_TAGS["primary"]),
        dict(identity, encoded_expert_ids=list(range(EXPERT_COUNT))),
        dict(identity, predecessor_content_key="c" * 64),
    ):
        with pytest.raises(AssertionError, match="stale or corrupt"):
            _validated_burn_cell(path, stale)


def test_burn_pass_tags_are_pinned_in_cell_identity_and_filename():
    identity = _burn_cell_identity(
        layer=0, projection="gate_proj",
        pass_tag=BURN_PASS_TAGS["scout"], rung=28,
        expert_ids=(0,), encoded_expert_ids=(0,),
        content_guard={"source_digest": "a" * 64},
        predecessor_content_key=None, replay=False,
    )
    assert identity["schema"] == BURN_CELL_IDENTITY_SCHEMA
    assert identity["pass_tag_schema"] == BURN_PASS_TAG_SCHEMA
    assert identity["pass_tag"] == "v2s-scout"
    with pytest.raises(ValueError, match="unregistered burn pass tag"):
        _burn_cell_path(0, "gate_proj", "v2-scout", 28)


def test_burn_resume_identity_mismatch_invalidates_dependent_suffix(
    tmp_path, monkeypatch,
):
    cells = tmp_path / "cells"
    burn = tmp_path / "burn"
    cells.mkdir()
    monkeypatch.setattr("tools.dsv4_afast_burn.BURN_CELL_ROOT", cells)
    monkeypatch.setattr("tools.dsv4_afast_burn.BURN_ROOT", burn)
    pass_tag = BURN_PASS_TAGS["scout"]
    expected = _burn_cell_identity(
        layer=0, projection="gate_proj", pass_tag=pass_tag, rung=32,
        expert_ids=(0,), encoded_expert_ids=(0,),
        content_guard={"source_digest": "b" * 64},
        predecessor_content_key="c" * 64, replay=False,
    )
    stale = dict(expected)
    stale["content_guard"] = {"source_digest": "a" * 64}
    for rung in (28, 32, 33, 38):
        path = _burn_cell_path(0, "gate_proj", pass_tag, rung)
        atomic_pickle(path, {
            "schema": BURN_CELL_SCHEMA,
            "pass_tag_schema": BURN_PASS_TAG_SCHEMA,
            "pass_tag": pass_tag,
            "content_key": _sha(stale),
            "identity": stale,
            "cell": {
                "rung": rung, "expert_ids": [0],
                "warm_state_path": str(tmp_path / "unused"),
            },
        })
    result = _validated_burn_cell_or_invalidate(
        _burn_cell_path(0, "gate_proj", pass_tag, 32), expected,
    )
    assert result is None
    assert _burn_cell_path(0, "gate_proj", pass_tag, 28).is_file()
    for rung in (32, 33, 38):
        assert not _burn_cell_path(0, "gate_proj", pass_tag, rung).exists()
    manifests = list(
        (burn / "quarantine-content-mismatch").glob("*/MANIFEST.json")
    )
    assert len(manifests) == 1
    evidence = json.loads(manifests[0].read_text())
    assert evidence["pass_tag_schema"] == BURN_PASS_TAG_SCHEMA
    assert [item["rung"] for item in evidence["moved"]] == [32, 33, 38]


def test_grid_cell_resume_is_content_keyed_and_requires_outputs(tmp_path):
    completion = tmp_path / "CELL_COMPLETE.json"
    required = tmp_path / "selection.json"
    required.write_text("{}")
    identity = {
        "schema": "prismaquant.dsv4_afast_grid_cell_identity.v1",
        "name": "b-92",
        "variant": "mtp-in",
    }
    result = complete_grid_cell(
        completion, identity, {"directory": str(tmp_path)},
    )
    assert result["content_key"]
    assert _validated_completion(completion, identity, (required,)) == result
    required.unlink()
    with pytest.raises(AssertionError, match="stale or corrupt"):
        _validated_completion(completion, identity, (required,))


def test_resume_mismatch_quarantines_only_dependent_projection_suffix(
    tmp_path, monkeypatch,
):
    cells = tmp_path / "cells"
    pilot = tmp_path / "pilot"
    cells.mkdir()
    monkeypatch.setattr(afast_campaign, "PILOT2_CELL_SHARDS", cells)
    monkeypatch.setattr(afast_campaign, "PILOT2_ROOT", pilot)
    for rung in (30, 31, 32):
        (cells / f"layer_014_gate_proj_K{rung}.pkl").write_bytes(
            f"K{rung}".encode()
        )
    (cells / "layer_014_up_proj_K32.pkl").write_bytes(b"other projection")
    manifest = _quarantine_projection_suffix(
        layer=14, projection="gate_proj", first_rung=31,
        mismatch={"reason": "unit-test"},
    )
    assert (cells / "layer_014_gate_proj_K30.pkl").is_file()
    assert not (cells / "layer_014_gate_proj_K31.pkl").exists()
    assert not (cells / "layer_014_gate_proj_K32.pkl").exists()
    assert (cells / "layer_014_up_proj_K32.pkl").is_file()
    evidence = json.loads(manifest.read_text())
    assert [item["rung"] for item in evidence["moved"]] == [31, 32]
