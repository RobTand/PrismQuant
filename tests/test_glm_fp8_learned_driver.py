from __future__ import annotations

import hashlib
import copy
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

    assert "load_active_glm_corpus_bound(" in source
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
    assert "saved_cell = saved_per_tensor.get(entry.name)" in source
    assert "_require_fp8_replay_match(entry.name, saved_cell, cell)" in source


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
        "environment": {"execution": "bound"},
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
        _DRIVER,
        "_execution_environment",
        lambda _ladder, require_cuda: (
            {"execution": "bound"}, {"segment": "bound"}
        ),
    )
    monkeypatch.setattr(
        _DRIVER, "_locked_sources", lambda _path: {"locked": "bound"}
    )
    monkeypatch.setattr(
        _DRIVER, "_frozen_codec_closure", lambda _ladder: {"closure": "bound"}
    )
    monkeypatch.setattr(
        _DRIVER, "load_active_glm_corpus_bound",
        lambda _root, _path: (
            fresh,
            {
                "path": str(manifest.resolve()),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
        ),
    )
    assert _DRIVER._verify_final_bindings(
        args=args, settings=settings, ladder=ladder
    ) == {"segment": "bound"}

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


def test_fp8_resume_requires_exact_regenerated_metrics_hashes_and_books():
    regenerated = {
        "weighted_energy": 10.0,
        "arms": {
            "fp8_cb@32": {
                "encode_seconds_observation_not_perf_claim": 2.0,
                "weighted_sse": 1.0,
                "weighted_nsse": 0.1,
                "weighted_snr_db": 10.0,
                "reconstruction_sha256": "a" * 64,
            },
            "fp8_cb_learned@32": {
                "encode_seconds_observation_not_perf_claim": 3.0,
                "weighted_sse": 0.5,
                "weighted_nsse": 0.05,
                "weighted_snr_db": 13.010299956639813,
                "reconstruction_sha256": "b" * 64,
                "learned_book": {
                    "elements": 16,
                    "tables": [{"sha256": "c" * 64}],
                },
            },
        },
    }
    timing_only = copy.deepcopy(regenerated)
    timing_only["arms"]["fp8_cb@32"][
        "encode_seconds_observation_not_perf_claim"
    ] = 999.0
    _DRIVER._require_fp8_replay_match(
        "tensor-a", timing_only, regenerated
    )

    attacks = []
    invented = copy.deepcopy(regenerated)
    invented["arms"]["fp8_cb@32"].update({
        "weighted_sse": 2.0,
        "weighted_nsse": 0.2,
        "weighted_snr_db": 6.9897000433601875,
    })
    attacks.append(invented)
    false_reconstruction = copy.deepcopy(regenerated)
    false_reconstruction["arms"]["fp8_cb@32"][
        "reconstruction_sha256"
    ] = "d" * 64
    attacks.append(false_reconstruction)
    false_book = copy.deepcopy(regenerated)
    false_book["arms"]["fp8_cb_learned@32"]["learned_book"]["tables"][0][
        "sha256"
    ] = "e" * 64
    attacks.append(false_book)
    for attack in attacks:
        with pytest.raises(_DRIVER.CampaignError, match="deterministic replay"):
            _DRIVER._require_fp8_replay_match(
                "tensor-a", attack, regenerated
            )


def test_dry_run_is_gpu_optional_and_cannot_construct_publication_receipt(
    tmp_path, monkeypatch, capsys,
):
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "result.json"
    ladder_path = tmp_path / "fp8_ladder.py"
    corpus = SimpleNamespace(
        manifest_path=manifest,
        manifest={
            "file_sha256": "f" * 64,
            "importance_identity": {"value_sha256": "i" * 64},
            "prismaquant_commit": "c" * 40,
        },
        populations={"dense": [object()], "routed": []},
    )
    monkeypatch.setattr(
        _DRIVER,
        "load_active_glm_corpus_bound",
        lambda _root, _path: (corpus, {"sha256": "m" * 64}),
    )
    monkeypatch.setattr(_DRIVER, "_locked_sources", lambda _path: {"locked": True})
    monkeypatch.setattr(
        _DRIVER,
        "_load_ladder",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("dry-run imported the GPU ladder")
        ),
    )
    monkeypatch.setattr(
        _DRIVER, "_frozen_codec_closure", lambda _ladder: {"closure": True}
    )
    monkeypatch.setattr(
        _DRIVER, "_active_source_identity", lambda: {"source": True}
    )
    monkeypatch.setattr(
        _DRIVER,
        "_execution_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run attempted publication attestation")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_PATH),
            "--manifest", str(manifest),
            "--locked-ladder", str(ladder_path),
            "--out", str(out),
            "--dry-run",
        ],
    )

    assert _DRIVER.main() == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["status"] == "validated_no_gpu_no_write"
    assert preflight["publication_capable"] is False
    assert preflight["publication_receipt"] is None
    assert "environment" not in preflight
    assert not out.exists()
