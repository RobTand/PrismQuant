from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/fp8_learned_glm.py"
)
_SPEC = importlib.util.spec_from_file_location("fp8_learned_glm", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DRIVER = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_PATH.parent))
try:
    _SPEC.loader.exec_module(_DRIVER)
finally:
    sys.path.pop(0)


def _cell(population, fixed, learned, fixed_bpw=4.0, learned_bpw=4.1):
    arms = {}
    for rung in _DRIVER.RUNGS:
        arms[f"fp8_cb@{rung}"] = {
            "weighted_snr_db": fixed + rung / 100,
            "footprint": {"exact_bpw": fixed_bpw + rung / 1000},
        }
        arms[f"fp8_cb_learned@{rung}"] = {
            "weighted_snr_db": learned + rung / 100,
            "footprint": {"exact_bpw": learned_bpw + rung / 1000},
        }
    return {"population": population, "arms": arms}


def test_population_summary_never_pools_dense_and_routed():
    summaries = _DRIVER.population_summaries({
        "dense-a": _cell("dense", 10.0, 11.0),
        "dense-b": _cell("dense", 12.0, 13.0),
        "routed-a": _cell("routed", 20.0, 24.0),
    })
    assert set(summaries) == {"dense", "routed"}
    assert summaries["dense"]["tensors"] == 2
    assert summaries["routed"]["tensors"] == 1
    assert all(row["learned_minus_fixed_db_median"] == 1.0
               for row in summaries["dense"]["rows"])
    assert all(row["learned_minus_fixed_db_median"] == 4.0
               for row in summaries["routed"]["rows"])
    assert "all" not in summaries and "pooled" not in summaries


def test_corpus_reader_does_not_claim_canonical_prismaquant_package():
    source = _PATH.read_text()

    assert "load_active_glm_corpus(REPO_ROOT, args.manifest)" in source
    assert "from prismaquant.trellis_bf16_corpus import" not in source


def test_future_v2_binds_transitive_sources_and_result_last_publication():
    source = _PATH.read_text()

    assert _DRIVER.SCHEMA == "trellis.glm_fp8_learned_balanced.v2"
    assert '"corpus_manifest_sha256"' in source
    assert '"corpus_file_sha256"' in source
    assert '"importance_value_sha256"' in source
    assert '"active_source_identity"' in source
    assert '"frozen_codec_closure"' in source
    assert "hull.snapshot_tree_sha256()" in source
    assert "hull.source_hashes()" in source
    assert "publish_file_no_replace(partial, args.out)" in source


def test_final_binding_recheck_refuses_midrun_corpus_mutation(
    tmp_path, monkeypatch,
):
    manifest = tmp_path / "manifest.json"
    artifact = tmp_path / "corpus.safetensors"
    manifest.write_text("manifest-v1\n")
    artifact.write_bytes(b"artifact-v1")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    fresh = SimpleNamespace(
        manifest_path=manifest,
        artifact_path=artifact,
        manifest={
            "file_sha256": artifact_hash,
            "importance_identity": {"value_sha256": "i" * 64},
            "prismaquant_commit": "c" * 40,
        },
    )
    settings = {
        "active_source_identity": {"active": "bound"},
        "locked_sources": {"locked": "bound"},
        "frozen_codec_closure": {"closure": "bound"},
        "corpus_manifest_sha256": hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
        "corpus_file_sha256": artifact_hash,
        "importance_value_sha256": "i" * 64,
        "corpus_prismaquant_commit": "c" * 40,
    }
    args = SimpleNamespace(
        locked_ladder=tmp_path / "fp8_ladder.py",
        manifest=manifest,
    )
    ladder = object()
    monkeypatch.setattr(
        _DRIVER, "_active_source_identity", lambda: {"active": "bound"}
    )
    monkeypatch.setattr(
        _DRIVER, "_locked_sources", lambda _path: {"locked": "bound"}
    )
    monkeypatch.setattr(
        _DRIVER, "_frozen_codec_closure", lambda _ladder: {"closure": "bound"}
    )
    monkeypatch.setattr(
        _DRIVER, "load_active_glm_corpus", lambda _root, _path: fresh
    )
    _DRIVER._verify_final_bindings(args=args, settings=settings, ladder=ladder)

    artifact.write_bytes(b"artifact-mutated-during-run")
    with pytest.raises(_DRIVER.CampaignError, match="corpus drifted"):
        _DRIVER._verify_final_bindings(args=args, settings=settings, ladder=ladder)


def test_fp8_self_bound_partial_checks_digest_before_closed_semantics(tmp_path):
    entry = SimpleNamespace(
        name="tensor-a",
        population="dense",
        source_weight_sha256="w" * 64,
        importance_sha256="i" * 64,
        source_weight_shape=(2, 3),
    )
    corpus = SimpleNamespace(entries=(entry,))
    settings = {"schema": _DRIVER.SCHEMA, "identity_sha256": "s" * 64}
    arms = {
        f"{family}@{rung}": {}
        for rung in _DRIVER.RUNGS
        for family in ("fp8_cb", "fp8_cb_learned")
    }
    report = {
        "schema": _DRIVER.SCHEMA,
        "settings": settings,
        "started_at_unix_s": 1.0,
        "per_tensor": {
            "tensor-a": {
                "population": "dense",
                "shape": [2, 3],
                "source_weight_sha256": "w" * 64,
                "importance_sha256": "i" * 64,
                "importance_source": {},
                "weighted_energy": 1.0,
                "arms": arms,
            }
        },
        "partial": True,
        "tensors_done": 1,
    }
    path = tmp_path / "fp8.partial"
    path.write_text(json.dumps(_DRIVER._sealed_report(report)))
    with pytest.raises(_DRIVER.CampaignError, match="contract differs"):
        _DRIVER._resume_report(path, settings=settings, corpus=corpus)

    mutated = json.loads(path.read_text())
    mutated["per_tensor"]["tensor-a"]["weighted_energy"] = 2.0
    path.write_text(json.dumps(mutated))
    with pytest.raises(_DRIVER.CampaignError, match="self-digest differs"):
        _DRIVER._resume_report(path, settings=settings, corpus=corpus)
