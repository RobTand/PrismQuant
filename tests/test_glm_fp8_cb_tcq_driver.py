from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/fp8_cb_tcq_glm.py"
)
_SPEC = importlib.util.spec_from_file_location("fp8_cb_tcq_glm", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DRIVER = importlib.util.module_from_spec(_SPEC)
import sys

sys.path.insert(0, str(_PATH.parent))
try:
    _SPEC.loader.exec_module(_DRIVER)
finally:
    sys.path.pop(0)


def _point(*, snr: float, bpw: float, learned_wire_bpw: float | None = None):
    arm = {
        "weighted_snr_db": snr,
        "footprint": {"exact_bpw": bpw},
    }
    if learned_wire_bpw is not None:
        arm["book_kind"] = "per_tensor_weighted_lloyd"
        arm["footprint"]["exact_bpw_book_wire8"] = learned_wire_bpw
    return arm


def _verdict_cell(population: str, *, production_tcq=11.5, penalty_tcq=11.5):
    arms = {
        "fp8_cb_fixed@32": _point(snr=10.0, bpw=4.0),
        "fp8_cb_learned@32": _point(
            snr=12.0, bpw=4.2, learned_wire_bpw=4.05
        ),
        "fp8_cb_fixed@40": _point(snr=15.0, bpw=5.0),
        "fp8_cb_learned@40": _point(
            snr=17.0, bpw=5.2, learned_wire_bpw=5.05
        ),
    }
    for bracket, best in (
        ("production_row_fp32", production_tcq),
        ("two_tier", penalty_tcq),
    ):
        for rate in (4, 5):
            arms[f"tcq_e4m3.{bracket}.lloyd@{rate}"] = _point(
                snr=best - 0.5, bpw=float(rate) + 0.1
            )
            arms[f"tcq_e4m3.{bracket}.exact_dp@{rate}"] = _point(
                snr=best, bpw=float(rate) + 0.1
            )
    return {"population": population, "arms": arms}


def test_learned_book_price_is_load_bearing_for_exact_byte_verdict():
    cell = _verdict_cell("dense")

    wire = _DRIVER._tensor_cell_verdict(
        cell, rate=4, bracket="production_row_fp32", book_price="wire8"
    )
    production = _DRIVER._tensor_cell_verdict(
        cell,
        rate=4,
        bracket="production_row_fp32",
        book_price="fp16_production",
    )

    assert wire["verdict"] == "FP8_CB"
    assert production["verdict"] == "NO_VERDICT_exact_byte_frontiers_cross"
    assert wire["best_quality_cb_bpw"] == 4.05
    assert production["best_quality_cb_bpw"] == 4.2


def test_population_summary_separates_populations_and_requires_all_brackets():
    # Production says TCQ while the penalty bracket says CB.  A vote or pooled
    # median would fabricate a conclusion; the contract must return no verdict.
    cells = {
        "dense-a": _verdict_cell("dense", production_tcq=18.0, penalty_tcq=9.0),
        "routed-a": _verdict_cell("routed", production_tcq=18.0, penalty_tcq=9.0),
        "routed-b": _verdict_cell("routed", production_tcq=18.0, penalty_tcq=9.0),
    }
    for cell in cells.values():
        for rate in (4, 5):
            for selector in _DRIVER.ALPHABET_SELECTORS:
                cell["arms"][
                    f"tcq_e4m3.production_row_fp32.{selector}@{rate}"
                ]["footprint"]["exact_bpw"] = rate - 0.1
    summary = _DRIVER.population_summaries(cells)

    assert set(summary) == {"dense", "routed"}
    assert summary["dense"]["tensors"] == 1
    assert summary["routed"]["tensors"] == 2
    assert "pooled" not in summary and "all" not in summary
    assert all(
        row["verdict"] == "NO_VERDICT_brackets_disagree_or_frontiers_cross"
        for population in summary.values()
        for row in population["cells"]
    )


def test_family_frontier_refuses_quality_only_win_at_more_bytes():
    candidate = _DRIVER._frontier([
        {"arm": "candidate", "bpw": 4.2, "snr": 20.0},
    ])
    incumbent = _DRIVER._frontier([
        {"arm": "incumbent", "bpw": 4.0, "snr": 10.0},
    ])

    assert not _DRIVER._family_dominates(candidate, incumbent)
    assert not _DRIVER._family_dominates(incumbent, candidate)


def _full_arm(name: str, *, numel: int = 512):
    total_bits = 2048
    arm = {
        "encode_seconds_observation_not_perf_claim": 1.0,
        "weighted_sse": 1.0,
        "weighted_nsse": 0.1,
        "weighted_snr_db": 10.0,
        "reconstruction_sha256": "a" * 64,
        "footprint": {
            "total_bits": total_bits,
            "exact_bpw": total_bits / numel,
        },
    }
    if name.startswith("fp8_cb_"):
        arm.update({
            "family": "FP8_CB_K",
            "rung": int(name.rsplit("@", 1)[1]),
            "encode_tier": _DRIVER.ENCODE_TIER,
            "book_kind": (
                "per_tensor_weighted_lloyd"
                if name.startswith("fp8_cb_learned@")
                else "fixed_lattice"
            ),
        })
        if name.startswith("fp8_cb_learned@"):
            arm["learned_book"] = {"test_fixture": True}
            arm["footprint"]["exact_bpw_book_wire8"] = 3.5
    else:
        stem, rate = name.rsplit("@", 1)
        _family, bracket, selector = stem.split(".")
        arm.update({
            "family": "TCQ_E4M3_R256",
            "rate": float(rate),
            "trellis_scale_bracket": bracket,
            "alphabet_selector": selector,
            "e4m3_plane_sha256": "e" * 64,
            "alphabet": {"test_fixture": True},
            "schedule": {"test_fixture": True},
        })
    return arm


def _complete_cell(population: str):
    return {
        "population": population,
        "shape": [2, 256],
        "source_weight_sha256": "b" * 64,
        "importance_sha256": "c" * 64,
        "importance_source": {
            "qname": "q",
            "expert": None,
            "denominator_name": "n_tokens_seen",
            "denominator": 1,
        },
        "metric_weight_sha256": "d" * 64,
        "weighted_energy": 10.0,
        "arms": {
            name: _full_arm(name) for name in _DRIVER.ARM_NAMES
        },
    }


def _entry(name: str, population: str):
    return SimpleNamespace(
        name=name,
        population=population,
        source_weight_shape=(2, 256),
        source_weight_sha256="b" * 64,
        importance_sha256="c" * 64,
        importance_source_qname="q",
        importance_source_expert=None,
        importance_denominator_name="n_tokens_seen",
        importance_denominator=1,
    )


def _settings(entries):
    commit = "e" * 40
    population_counts = {}
    for entry in entries:
        population_counts[entry.population] = (
            population_counts.get(entry.population, 0) + 1
        )
    settings = {
        "schema": _DRIVER.SCHEMA,
        "corpus_manifest": "/immutable/manifest.json",
        "corpus_manifest_sha256": "1" * 64,
        "corpus_file_sha256": "2" * 64,
        "importance_value_sha256": "3" * 64,
        "corpus_prismaquant_commit": "4" * 40,
        "population_counts": population_counts,
        "rungs": list(_DRIVER.RUNGS),
        "rates": list(_DRIVER.RATES),
        "cell_map": {
            str(rate): rung for rate, rung in _DRIVER.CELL_MAP.items()
        },
        "trellis_scale_brackets": list(_DRIVER.TRELLIS_BRACKETS),
        "alphabet_selectors": list(_DRIVER.ALPHABET_SELECTORS),
        "book_price_brackets": list(_DRIVER.BOOK_PRICE_BRACKETS),
        "encode_tier": _DRIVER.ENCODE_TIER,
        "locked_sources": {},
        "frozen_codec_closure": {},
        "active_source_identity": {
            "repo_git_commit": commit,
            "repo_root": "/immutable/prismaquant",
        },
        "environment": {
            "schema": "trellis.numeric_execution.v2",
            "physical_host": "sparky",
            "uts_hostname": "sparky",
            "gpu_uuid": "GPU-e76c7efc-c157-b1f4-1348-83e4eb5092f4",
            "container_image_reference": (
                "eugr/spark-vllm@sha256:"
                "58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869"
            ),
            "container_image_digest": (
                "sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869"
            ),
            "container_image_id": (
                "sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869"
            ),
            "container_image_evidence": (
                "host_docker_daemon_inspect_before_start"
            ),
            "container_image_in_process_verification": "not_available",
            "container_user": "1000:1000",
            "ipc_mode": "private",
            "repo_root": "/immutable/prismaquant",
            "source_mount_evidence": (
                "host_docker_daemon_inspect_readonly_repo_and_git"
            ),
            "repo_git_commit": commit,
            "repo_tree_clean": True,
            "python": "3.12.3",
            "torch": "2.13.0+cu130",
            "triton": "3.7.1",
            "device": "NVIDIA GB10",
        },
        "command": ["fp8_cb_tcq_glm.py"],
        "claim_boundary": _DRIVER.CLAIM_BOUNDARY,
    }
    settings["identity_sha256"] = _DRIVER.identity_sha256(settings)
    return settings


def _segment(settings, *, marker="1"):
    environment = settings["environment"]
    unsigned = {
        "schema": "trellis.numeric_execution_segment.v1",
        "physical_host": environment["physical_host"],
        "container_id": marker * 64,
        "image_id": environment["container_image_id"],
        "gpu_uuid": environment["gpu_uuid"],
        "launch_attestation_path": (
            f"/home/rob/dq-runs/test-fixtures/{marker}/attestation.json"
        ),
        "launch_attestation_sha256": "a" * 64,
        "launch_command_sha256": "b" * 64,
    }
    return {
        **unsigned,
        "segment_sha256": hashlib.sha256(json.dumps(
            unsigned, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
    }


def _evidence(per_tensor):
    return {
        name: _DRIVER._replay_semantics(cell)
        for name, cell in per_tensor.items()
    }


def test_closed_final_report_rederives_summaries_and_refuses_pooled_field():
    entries = (_entry("dense-a", "dense"), _entry("routed-a", "routed"))
    settings = _settings(entries)
    per_tensor = {
        "dense-a": _complete_cell("dense"),
        "routed-a": _complete_cell("routed"),
    }
    report = {
        "schema": _DRIVER.SCHEMA,
        "settings": settings,
        "started_at_unix_s": 1.0,
        "per_tensor": per_tensor,
        "partial": False,
        "tensors_done": 2,
        "execution_segments": [_segment(settings)],
        "completed_at_unix_s": 2.0,
        "population_summaries": _DRIVER.population_summaries(per_tensor),
        "status": _DRIVER.STATUS,
        "claim_boundary": _DRIVER.CLAIM_BOUNDARY,
    }
    sealed = _DRIVER._sealed(report)
    _DRIVER.validate_report(
        sealed, settings=settings, entries=entries, require_complete=True,
        generated_evidence=_evidence(per_tensor),
    )

    attack = copy.deepcopy(sealed)
    attack["pooled_summary"] = {"winner": "TCQ"}
    attack = _DRIVER._sealed(attack)
    with pytest.raises(_DRIVER.CampaignError, match="unknown=.*pooled_summary"):
        _DRIVER.validate_report(
            attack, settings=settings, entries=entries, require_complete=True,
            generated_evidence=_evidence(per_tensor),
        )


def test_resume_requires_full_claim_replay_but_ignores_wall_timing():
    saved = _complete_cell("dense")
    regenerated = copy.deepcopy(saved)
    regenerated["arms"][_DRIVER.ARM_NAMES[0]][
        "encode_seconds_observation_not_perf_claim"
    ] = 999.0
    _DRIVER.require_replay_match("tensor", saved, regenerated)

    regenerated["arms"][_DRIVER.ARM_NAMES[0]]["weighted_sse"] = 2.0
    with pytest.raises(_DRIVER.CampaignError, match="full replay"):
        _DRIVER.require_replay_match("tensor", saved, regenerated)


def test_closed_report_binds_each_arm_to_its_name():
    entries = (_entry("dense-a", "dense"), _entry("routed-a", "routed"))
    settings = _settings(entries)
    per_tensor = {
        "dense-a": _complete_cell("dense"),
        "routed-a": _complete_cell("routed"),
    }
    report = {
        "schema": _DRIVER.SCHEMA,
        "settings": settings,
        "started_at_unix_s": 1.0,
        "per_tensor": per_tensor,
        "partial": False,
        "tensors_done": 2,
        "execution_segments": [_segment(settings)],
        "completed_at_unix_s": 2.0,
        "population_summaries": _DRIVER.population_summaries(per_tensor),
        "status": _DRIVER.STATUS,
        "claim_boundary": _DRIVER.CLAIM_BOUNDARY,
    }
    report["per_tensor"]["dense-a"]["arms"][
        "tcq_e4m3.two_tier.exact_dp@5"
    ]["alphabet_selector"] = "lloyd"
    sealed = _DRIVER._sealed(report)

    with pytest.raises(_DRIVER.CampaignError, match="TCQ identity differs"):
        _DRIVER.validate_report(
            sealed, settings=settings, entries=entries, require_complete=True,
            generated_evidence=_evidence(per_tensor),
        )


def test_closed_report_binds_importance_provenance_to_corpus_entry():
    entries = (_entry("dense-a", "dense"), _entry("routed-a", "routed"))
    settings = _settings(entries)
    per_tensor = {
        "dense-a": _complete_cell("dense"),
        "routed-a": _complete_cell("routed"),
    }
    report = {
        "schema": _DRIVER.SCHEMA,
        "settings": settings,
        "started_at_unix_s": 1.0,
        "per_tensor": per_tensor,
        "partial": False,
        "tensors_done": 2,
        "execution_segments": [_segment(settings)],
        "completed_at_unix_s": 2.0,
        "population_summaries": _DRIVER.population_summaries(per_tensor),
        "status": _DRIVER.STATUS,
        "claim_boundary": _DRIVER.CLAIM_BOUNDARY,
    }
    report["per_tensor"]["dense-a"]["importance_source"]["qname"] = "wrong"
    sealed = _DRIVER._sealed(report)

    with pytest.raises(_DRIVER.CampaignError, match="importance provenance differs"):
        _DRIVER.validate_report(
            sealed, settings=settings, entries=entries, require_complete=True,
            generated_evidence=_evidence(per_tensor),
        )


def test_resealed_schedule_alphabet_book_and_reconstruction_mutations_fail():
    entries = (_entry("dense-a", "dense"), _entry("routed-a", "routed"))
    settings = _settings(entries)
    per_tensor = {
        "dense-a": _complete_cell("dense"),
        "routed-a": _complete_cell("routed"),
    }
    evidence = _evidence(per_tensor)
    report = {
        "schema": _DRIVER.SCHEMA,
        "settings": settings,
        "started_at_unix_s": 1.0,
        "per_tensor": per_tensor,
        "partial": False,
        "tensors_done": 2,
        "execution_segments": [_segment(settings)],
        "completed_at_unix_s": 2.0,
        "population_summaries": _DRIVER.population_summaries(per_tensor),
        "status": _DRIVER.STATUS,
        "claim_boundary": _DRIVER.CLAIM_BOUNDARY,
    }
    _DRIVER.validate_report(
        _DRIVER._sealed(report), settings=settings, entries=entries,
        require_complete=True, generated_evidence=evidence,
    )

    mutations = (
        lambda value: value["per_tensor"]["dense-a"]["arms"][
            "tcq_e4m3.two_tier.exact_dp@5"
        ]["schedule"].__setitem__("test_fixture", False),
        lambda value: value["per_tensor"]["dense-a"]["arms"][
            "tcq_e4m3.two_tier.exact_dp@5"
        ]["alphabet"].__setitem__("test_fixture", False),
        lambda value: value["per_tensor"]["dense-a"]["arms"][
            "fp8_cb_learned@32"
        ]["learned_book"].__setitem__("test_fixture", False),
        lambda value: value["per_tensor"]["dense-a"]["arms"][
            "fp8_cb_fixed@32"
        ].__setitem__("reconstruction_sha256", "0" * 64),
    )
    for mutate in mutations:
        bad = copy.deepcopy(report)
        mutate(bad)
        with pytest.raises(_DRIVER.CampaignError, match="generated evidence"):
            _DRIVER.validate_report(
                _DRIVER._sealed(bad), settings=settings, entries=entries,
                require_complete=True, generated_evidence=evidence,
            )

    bad_time = copy.deepcopy(report)
    bad_time["completed_at_unix_s"] = 0.5
    with pytest.raises(_DRIVER.CampaignError, match="completion precedes start"):
        _DRIVER.validate_report(
            _DRIVER._sealed(bad_time), settings=settings, entries=entries,
            require_complete=True, generated_evidence=evidence,
        )


def test_fresh_container_segment_history_is_append_only_and_deduplicated():
    entries = (_entry("dense-a", "dense"), _entry("routed-a", "routed"))
    settings = _settings(entries)
    per_tensor = {
        "dense-a": _complete_cell("dense"),
        "routed-a": _complete_cell("routed"),
    }
    report = {
        "schema": _DRIVER.SCHEMA,
        "settings": settings,
        "started_at_unix_s": 1.0,
        "per_tensor": per_tensor,
        "partial": False,
        "tensors_done": 2,
        "execution_segments": [_segment(settings), _segment(settings, marker="2")],
        "completed_at_unix_s": 2.0,
        "population_summaries": _DRIVER.population_summaries(per_tensor),
        "status": _DRIVER.STATUS,
        "claim_boundary": _DRIVER.CLAIM_BOUNDARY,
    }
    evidence = _evidence(per_tensor)
    _DRIVER.validate_report(
        _DRIVER._sealed(report), settings=settings, entries=entries,
        require_complete=True, generated_evidence=evidence,
    )
    report["execution_segments"].append(copy.deepcopy(report["execution_segments"][0]))
    with pytest.raises(_DRIVER.CampaignError, match="duplicate"):
        _DRIVER.validate_report(
            _DRIVER._sealed(report), settings=settings, entries=entries,
            require_complete=True, generated_evidence=evidence,
        )


def test_driver_is_cuda_only_and_publishes_result_last():
    source = _PATH.read_text()

    assert "this GPU campaign has no CPU fallback" in source
    assert "require_numeric_execution_environment(" in source
    assert '"numeric_execution_contract"' in source
    assert "_driver_environment" not in source
    assert "--expected-host" not in source
    assert "--container-identity" not in source
    assert 'backend="triton"' in source
    assert "validate_report(" in source
    assert source.index("_verify_final_bindings(") < source.rindex(
        "publish_file_no_replace(partial, args.out)"
    )
    assert _DRIVER.CLAIM_BOUNDARY["serving_verdict"] is False
    assert _DRIVER.CLAIM_BOUNDARY["performance_claim"] is False


def test_settings_contract_closes_execution_identity_and_receipt_fields():
    entries = (_entry("dense-a", "dense"), _entry("routed-a", "routed"))
    settings = _settings(entries)
    _DRIVER._validate_settings(settings, entries)

    attacks = []
    extra = copy.deepcopy(settings)
    extra["declared_container_identity"] = extra["environment"][
        "container_image_digest"
    ]
    attacks.append(extra)
    wrong_gpu = copy.deepcopy(settings)
    wrong_gpu["environment"]["gpu_uuid"] = (
        "GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705"
    )
    attacks.append(wrong_gpu)
    overclaim = copy.deepcopy(settings)
    overclaim["environment"]["container_image_in_process_verification"] = (
        "cryptographic"
    )
    attacks.append(overclaim)
    resume_mismatch = copy.deepcopy(settings)
    resume_mismatch["environment"]["container_image_id"] = "sha256:" + "6" * 64
    attacks.append(resume_mismatch)
    for attack in attacks:
        with pytest.raises(_DRIVER.CampaignError):
            _DRIVER._validate_settings(attack, entries)


def test_final_binding_recheck_refuses_execution_environment_drift(
    tmp_path, monkeypatch,
):
    artifact = tmp_path / "corpus.safetensors"
    artifact.write_bytes(b"corpus")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"manifest")
    corpus = SimpleNamespace(
        artifact_path=artifact,
        manifest={
            "importance_identity": {"value_sha256": "i" * 64},
            "prismaquant_commit": "c" * 40,
        },
    )
    settings = {
        "environment": {"execution": "bound"},
        "active_source_identity": {"active": "bound"},
        "locked_sources": {"locked": "bound"},
        "frozen_codec_closure": {"closure": "bound"},
        "corpus_manifest_sha256": _DRIVER.file_sha256(manifest),
        "corpus_file_sha256": _DRIVER.file_sha256(artifact),
        "importance_value_sha256": "i" * 64,
        "corpus_prismaquant_commit": "c" * 40,
    }
    args = SimpleNamespace(locked_ladder=tmp_path / "ladder.py", manifest=manifest)
    live = {"execution": "bound"}
    segment = {"segment": "bound"}
    monkeypatch.setattr(
        _DRIVER, "_execution_environment", lambda _ladder: (live, segment)
    )
    monkeypatch.setattr(
        _DRIVER, "_active_source_identity", lambda: {"active": "bound"}
    )
    monkeypatch.setattr(
        _DRIVER, "_locked_sources", lambda _path: {"locked": "bound"}
    )
    monkeypatch.setattr(
        _DRIVER.BASE,
        "_frozen_codec_closure",
        lambda _ladder: {"closure": "bound"},
    )
    monkeypatch.setattr(
        _DRIVER,
        "load_active_glm_corpus_bound",
        lambda _root, _manifest: (
            corpus,
            {"sha256": _DRIVER.file_sha256(manifest)},
        ),
    )
    assert _DRIVER._verify_final_bindings(
        args=args, settings=settings, ladder=object()
    ) == segment
    live = {"execution": "drifted"}
    with pytest.raises(_DRIVER.CampaignError, match="environment drifted"):
        _DRIVER._verify_final_bindings(
            args=args, settings=settings, ladder=object()
        )


def test_locked_contract_contains_both_cells_selectors_and_price_brackets():
    assert _DRIVER.RUNGS == (32, 40)
    assert _DRIVER.RATES == (4.0, 5.0)
    assert _DRIVER.TRELLIS_BRACKETS == (
        "production_row_fp32",
        "two_tier",
    )
    assert _DRIVER.ALPHABET_SELECTORS == ("lloyd", "exact_dp")
    assert _DRIVER.BOOK_PRICE_BRACKETS == ("wire8", "fp16_production")
    assert len(_DRIVER.ARM_NAMES) == 12


def test_preflight_is_gpu_optional_nonpublishing_and_needs_no_self_claims(
    tmp_path, monkeypatch, capsys,
):
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "result.json"
    ladder_path = tmp_path / "fp8_ladder.py"
    corpus = SimpleNamespace(
        manifest={"file_sha256": "f" * 64},
        populations={"dense": [object()], "routed": []},
    )
    monkeypatch.setattr(
        _DRIVER,
        "load_active_glm_corpus_bound",
        lambda _root, _path: (corpus, {"sha256": "m" * 64}),
    )
    monkeypatch.setattr(_DRIVER, "_locked_sources", lambda _path: {"locked": True})
    monkeypatch.setattr(
        _DRIVER.BASE,
        "_load_ladder",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("preflight imported the GPU ladder")
        ),
    )
    monkeypatch.setattr(
        _DRIVER, "_active_source_identity", lambda: {"source": True}
    )
    monkeypatch.setattr(
        _DRIVER,
        "_execution_environment",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("preflight attempted publication attestation")
        ),
    )

    assert _DRIVER.main([
        "--manifest", str(manifest),
        "--locked-ladder", str(ladder_path),
        "--out", str(out),
        "--preflight-only",
    ]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["status"] == "validated_no_gpu_no_write"
    assert preflight["publication_capable"] is False
    assert preflight["publication_receipt"] is None
    assert "environment" not in preflight
    assert not out.exists()
