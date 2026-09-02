from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from prismaquant.gridbook_serving_runtime_pin import (
    GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
)

import prismaquant.dspark_matched_performance as perf


TARGET_SHA = "a" * 64
DRAFT_SHA = "b" * 64
RUNTIME_PIN = {
    "schema": "prismaquant.gridbook_serving_runtime_pin.v1",
    "repository": "https://github.com/RobTand/gridbook.git",
    "commit": "c" * 40,
    "version": GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
    "version_is_release": True,
    "wheel_sha256": "d" * 64,
    "runtime_contract_schema": "gridbook.runtime-contract.v4",
    "required_abi_features": {
        "routed_moe_per_role_codebook_lut": 1,
        "source_fp8_block128_w8a16": 1,
        "dspark_construction_physical_bridge": 1,
    },
}


def _workload() -> dict:
    prompt_sha = [f"{index:064x}" for index in range(8)]
    return {
        "schema": perf.WORKLOAD_SCHEMA,
        "prompt_count": 8,
        "prompt_sha256": prompt_sha,
        "prompt_corpus_sha256": perf.canonical_sha256(prompt_sha),
        "max_tokens_per_prompt": 128,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "warmup_repetitions": 1,
        "measured_repetitions": 1,
        "max_concurrency": 1,
    }


def _responses(workload: dict) -> list[dict]:
    return [
        {
            "index": index,
            "prompt_sha256": workload["prompt_sha256"][index],
            "content_sha256": f"{100 + index:064x}",
            "finish_reason": "length",
            "prompt_tokens": 20 + index,
            "completion_tokens": 128,
            "request_seconds": 1.0,
        }
        for index in range(8)
    ]


def _acceptance(workload: dict) -> dict:
    return {
        "schema": "prismaquant.dspark_acceptance_suite.v1",
        "started_at": "2026-08-13T00:00:05Z",
        "finished_at": "2026-08-13T00:00:07Z",
        "served_model": "dsv4-flash-gridbook-" + "1" * 32,
        "responses": _responses(workload),
        "acceptance": {
            "per_position_rates": [0.7, 0.5, 0.3, 0.2, 0.1],
        },
    }


def _routes() -> dict:
    return {
        "schema": "prismaquant.dspark_route_census.v1",
        "profile_sha256": "e" * 64,
        "serve_log_sha256": "f" * 64,
        "gemv_mode": "v2",
        "counts": {
            "target_fp4_v2": 70,
            "draft_fp4_v2": 6,
            "fp8_inherited": 16,
            "fallback": 0,
            "unscoped": 0,
            "total": 92,
        },
        "routes": [],
        "routes_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
    }


def _binding(model_sha: str, launch_model: str) -> dict:
    return {
        "schema": "prismaquant.served_artifact_binding/1",
        "resolved_path": launch_model,
        "launch_model": launch_model,
        "model_sha": model_sha,
        "artifact_inventory_sha256": "3" * 64,
        "artifact_bytes": 1024,
    }


def _graph(arm: str) -> dict:
    return {
        "serve_log_sha256": "4" * 64,
        "capture_marker": "Graph capturing finished in 1 secs, took -0.10 GiB",
        "capture_sizes": [1] if arm == perf.NO_MTP_ARM else [5, 6],
    }


def _memory(arm: str) -> dict:
    gib = perf.GIB
    rows = [
        ("startup", "2026-08-13T00:00:00Z", 120 * gib),
        ("ready", "2026-08-13T00:00:01Z", 10 * gib),
        ("warmup", "2026-08-13T00:00:03Z", 9 * gib),
        ("warmup", "2026-08-13T00:00:04Z", 9 * gib),
        ("measured", "2026-08-13T00:00:05Z", 9 * gib),
        ("measured", "2026-08-13T00:00:06Z", 9 * gib),
        ("measured", "2026-08-13T00:00:07Z", 9 * gib),
        ("post", "2026-08-13T00:00:08Z", 10 * gib),
    ]
    samples = [
        {
            "sequence": index,
            "observed_at": observed,
            "phase": phase,
            "mem_available_bytes": available,
        }
        for index, (phase, observed, available) in enumerate(rows)
    ]
    return {
        "schema": perf.MEMORY_SCHEMA,
        "memory_kind": "nvidia_gb10_unified",
        "sampler": "/proc/meminfo:MemAvailable",
        "sample_interval_ms": perf.MEMORY_SAMPLE_INTERVAL_MS,
        "samples": samples,
        "start_mem_available_bytes": 120 * gib,
        "ready_mem_available_bytes": 10 * gib,
        "startup_min_mem_available_bytes": 10 * gib,
        "measured_min_mem_available_bytes": 9 * gib,
        "post_mem_available_bytes": 10 * gib,
        "minimum_mem_available_bytes": 9 * gib,
        "model_residency_bytes": 110 * gib,
        "startup_transient_bytes": 0,
        "measured_transient_bytes": 1 * gib,
        "watchdog_floor_bytes": 4 * gib,
        "watchdog_tripped": False,
        "oom_events": 0,
        "oom_kill_detected": False,
        "server_alive_after": True,
        "kv_cache": {
            "schema": perf.KV_CAPACITY_SCHEMA,
            "dtype": "fp8",
            "requested_bytes": perf.KV_CACHE_MEMORY_BYTES,
            "allocated_bytes": perf.KV_CACHE_MEMORY_BYTES,
            "capacity_tokens": perf.MODEL_LEN,
            "max_model_len": perf.MODEL_LEN,
            "max_num_seqs": perf.MAX_NUM_SEQS,
            "max_num_batched_tokens": perf.MAX_NUM_BATCHED_TOKENS,
            "concurrency_at_max_model_len": 1.0,
            "profile_log_sha256": _graph(arm)["serve_log_sha256"],
            "capacity_verified": True,
        },
    }


def _manifest(arm: str, phase: str) -> dict:
    draft = _binding(DRAFT_SHA, "/draft") if arm == perf.MTP_ARM else None
    return {
        "attestation_phase": phase,
        "created": (
            "2026-08-13T00:00:02Z"
            if phase == "pre" else "2026-08-13T00:00:09Z"
        ),
        "serve_session_id": (
            "5" * 64 if arm == perf.NO_MTP_ARM else "6" * 64
        ),
        "serve_fingerprint": "7" * 64,
        "performance_stack_fingerprint": "8" * 64,
        "processes": [{"pid": 1}],
        "artifact_binding": _binding(TARGET_SHA, "/model"),
        "draft_artifact_binding": draft,
        "gridbook_runtime_pin": RUNTIME_PIN,
        "dspark_serving_profile": {"receipt_sha256": "d" * 64},
        "dspark_runtime_evidence": {"evidence_sha256": "e" * 64},
        "image": perf.DSV4_SPARK_VLLM_IMAGE,
        "gpu_name": "NVIDIA GB10",
        "gpu_uuid": "GPU-test",
        "gpu_count": 1,
        "driver_version": "test",
        "host_identity": {
            "boot_id": "boot",
            "machine_id_sha256": "9" * 64,
        },
        "package_versions": {
            "gridbook": "0.8.6",
            "vllm": perf.DSV4_SPARK_VLLM_VERSION,
        },
        "gridbook_distribution": {"sha256": "a" * 64},
        "resident_extensions": ["prismaquant_cb_ext.so"],
        "residency_readable": True,
        "server_process_environment": {
            "values": {"PRISMAQUANT_CB_GEMV": "v2"}
        },
    }


def _mtp_binding(acceptance: dict, routes: dict) -> dict:
    return {
        "schema": perf.MTP_BINDING_SCHEMA,
        "draft_model_sha": DRAFT_SHA,
        "draft_artifact_binding": _binding(DRAFT_SHA, "/draft"),
        "draft_format": "NVFP4_CB_K12",
        "num_speculative_tokens": 5,
        "acceptance": acceptance,
        "routes": routes,
        "routes_sha256": perf.canonical_sha256(routes),
    }


def _report(arm: str, workload: dict, acceptance: dict, routes: dict) -> dict:
    tps = 512.0 if arm == perf.NO_MTP_ARM else 550.0
    report = {
        "schema": perf.REPORT_SCHEMA,
        "arm": arm,
        "served_model": acceptance["served_model"],
        "started_at": "2026-08-13T00:00:05Z",
        "finished_at": "2026-08-13T00:00:07Z",
        "tool": perf.collector_tool_identity(
            source_root=Path(perf.__file__).resolve().parents[1],
            git_commit="b" * 40,
        ),
        "policy": perf.release_policy(predeclared_at="2026-08-12T00:00:00Z"),
        "target_model_sha": TARGET_SHA,
        "pre_manifest": _manifest(arm, "pre"),
        "post_manifest": _manifest(arm, "post"),
        "graph_capture": _graph(arm),
        "workload": workload,
        "warmup_responses": _responses(workload),
        "responses": _responses(workload),
        "counters": {
            "before": {key: 100 for key in perf._COUNTER_KEYS},
            "after": {
                "completed_requests": 108,
                "generation_tokens": 1124,
                "failed_requests": 100,
                "timed_out_requests": 100,
            },
            "delta": {
                "completed_requests": 8,
                "generation_tokens": 1024,
                "failed_requests": 0,
                "timed_out_requests": 0,
            },
        },
        "warm_decode": {
            "output_tokens": 1024,
            "wall_seconds": 2.0,
            "output_tokens_per_second": 512.0,
        },
        "memory": _memory(arm),
        "mtp": (
            _mtp_binding(acceptance, routes) if arm == perf.MTP_ARM else None
        ),
    }
    # Throughput is derived from the wall, so the synthetic MTP win uses a
    # shorter exact interval rather than writing an irreconcilable summary.
    if tps != 512.0:
        wall = 1024.0 / tps
        report["finished_at"] = "2026-08-13T00:00:06.861818Z"
        report["warm_decode"] = {
            "output_tokens": 1024,
            "wall_seconds": wall,
            "output_tokens_per_second": tps,
        }
        # The MTP report must be the exact acceptance interval.
        acceptance["finished_at"] = report["finished_at"]
    report["report_sha256"] = perf.canonical_sha256(report)
    return report


def _write_report(tmp_path, arm: str, report: dict):
    path = tmp_path / f"{arm}.json"
    path.write_text(json.dumps(report, sort_keys=True) + "\n")
    return path


def _evidence_pair(tmp_path, monkeypatch):
    workload = _workload()
    acceptance = _acceptance(workload)
    routes = _routes()
    no_report = _report(perf.NO_MTP_ARM, workload, acceptance, routes)
    mtp_report = _report(perf.MTP_ARM, workload, acceptance, routes)
    # The MTP report changed the shared acceptance finish; restamp its binding.
    mtp_report["mtp"]["acceptance"] = acceptance
    mtp_report["report_sha256"] = perf.canonical_sha256(
        {key: value for key, value in mtp_report.items() if key != "report_sha256"}
    )
    no_path = _write_report(tmp_path, perf.NO_MTP_ARM, no_report)
    mtp_path = _write_report(tmp_path, perf.MTP_ARM, mtp_report)
    monkeypatch.setattr(
        perf,
        "validate_dspark_serve_manifest",
        lambda manifest, **_kwargs: manifest["serve_fingerprint"],
    )
    common = {
        "expected_target_sha": TARGET_SHA,
        "expected_draft_sha": DRAFT_SHA,
        "expected_workload": workload,
        "expected_acceptance": acceptance,
        "expected_routes": routes,
        "requires_moe_marlin": True,
    }
    no_evidence = perf.validate_arm_report(
        no_report,
        report_path=no_path,
        arm=perf.NO_MTP_ARM,
        expected_speculative_config=None,
        expected_launch_options={"--compilation-config": "no-mtp-graph"},
        expected_launch_switches=set(),
        expected_no_mtp_graph_capture=no_report["graph_capture"],
        **common,
    )
    mtp_evidence = perf.validate_arm_report(
        mtp_report,
        report_path=mtp_path,
        arm=perf.MTP_ARM,
        expected_speculative_config={"method": "dspark"},
        expected_launch_options={
            "--compilation-config": "mtp-graph",
            "--speculative-config": "dspark-k5",
        },
        expected_launch_switches=set(),
        expected_mtp_pre_manifest=mtp_report["pre_manifest"],
        expected_mtp_post_manifest=mtp_report["post_manifest"],
        expected_mtp_graph_capture=mtp_report["graph_capture"],
        **common,
    )
    return workload, acceptance, routes, no_evidence, mtp_evidence


def _comparison_kwargs() -> dict:
    return {
        "expected_no_mtp_launch_options": {
            "--compilation-config": "no-mtp-graph"
        },
        "expected_mtp_launch_options": {
            "--compilation-config": "mtp-graph",
            "--speculative-config": "dspark-k5",
        },
        "expected_launch_switches": set(),
        "expected_mtp_speculative_config": {"method": "dspark"},
    }


def test_reports_bind_exact_arms_and_replay_strict_non_regression(
    tmp_path, monkeypatch
):
    workload, acceptance, routes, no_evidence, mtp_evidence = _evidence_pair(
        tmp_path, monkeypatch
    )
    result = perf.build_matched_result(
        no_mtp_evidence=no_evidence,
        mtp_evidence=mtp_evidence,
        target_model_sha=TARGET_SHA,
        draft_model_sha=DRAFT_SHA,
        runtime_pin=RUNTIME_PIN,
        expected_workload=workload,
        expected_acceptance=acceptance,
        expected_routes=routes,
        **_comparison_kwargs(),
    )
    assert result["comparison"]["mtp_to_no_mtp_ratio"] > 1.0
    assert perf.validate_matched_result(
        result,
        expected_target_sha=TARGET_SHA,
        expected_draft_sha=DRAFT_SHA,
        expected_runtime_pin=RUNTIME_PIN,
        expected_workload=workload,
        expected_acceptance=acceptance,
        expected_routes=routes,
        **_comparison_kwargs(),
    ) == result


def test_slow_mtp_mismatched_stack_and_low_headroom_fail(tmp_path, monkeypatch):
    workload, acceptance, routes, no_evidence, mtp_evidence = _evidence_pair(
        tmp_path, monkeypatch
    )
    faster_no_mtp = deepcopy(no_evidence)
    faster_no_mtp["warm_decode"] = {
        "output_tokens": 1024,
        "wall_seconds": 1024.0 / 600.0,
        "output_tokens_per_second": 600.0,
    }
    faster_no_mtp["finished_at"] = "2026-08-13T00:00:06.706667Z"
    faster_no_mtp["evidence_sha256"] = perf.canonical_sha256(
        {
            key: value
            for key, value in faster_no_mtp.items()
            if key != "evidence_sha256"
        }
    )
    with pytest.raises(perf.DSparkMatchedPerformanceError, match="regresses"):
        perf.build_matched_result(
            no_mtp_evidence=faster_no_mtp,
            mtp_evidence=mtp_evidence,
            target_model_sha=TARGET_SHA,
            draft_model_sha=DRAFT_SHA,
            runtime_pin=RUNTIME_PIN,
            expected_workload=workload,
            expected_acceptance=acceptance,
            expected_routes=routes,
            **_comparison_kwargs(),
        )

    mismatch = deepcopy(mtp_evidence)
    mismatch["manifest_identity"]["image"] = "wrong"
    mismatch["evidence_sha256"] = perf.canonical_sha256(
        {key: value for key, value in mismatch.items() if key != "evidence_sha256"}
    )
    with pytest.raises(perf.DSparkMatchedPerformanceError, match="image, runtime"):
        perf.build_matched_result(
            no_mtp_evidence=no_evidence,
            mtp_evidence=mismatch,
            target_model_sha=TARGET_SHA,
            draft_model_sha=DRAFT_SHA,
            runtime_pin=RUNTIME_PIN,
            expected_workload=workload,
            expected_acceptance=acceptance,
            expected_routes=routes,
            **_comparison_kwargs(),
        )

    low = deepcopy(no_evidence)
    low["memory"]["ready_mem_available_bytes"] = 7 * perf.GIB
    low["evidence_sha256"] = perf.canonical_sha256(
        {key: value for key, value in low.items() if key != "evidence_sha256"}
    )
    with pytest.raises(
        perf.DSparkMatchedPerformanceError,
        match="does not replay from its phase receipt",
    ):
        perf.build_matched_result(
            no_mtp_evidence=low,
            mtp_evidence=mtp_evidence,
            target_model_sha=TARGET_SHA,
            draft_model_sha=DRAFT_SHA,
            runtime_pin=RUNTIME_PIN,
            expected_workload=workload,
            expected_acceptance=acceptance,
            expected_routes=routes,
            **_comparison_kwargs(),
        )

    wrong_budget = deepcopy(mtp_evidence)
    wrong_budget["launch_contract"]["options"]["--max-model-len"] = "131072"
    wrong_budget["evidence_sha256"] = perf.canonical_sha256({
        key: value
        for key, value in wrong_budget.items()
        if key != "evidence_sha256"
    })
    with pytest.raises(
        perf.DSparkMatchedPerformanceError, match="launch contract differs"
    ):
        perf.build_matched_result(
            no_mtp_evidence=no_evidence,
            mtp_evidence=wrong_budget,
            target_model_sha=TARGET_SHA,
            draft_model_sha=DRAFT_SHA,
            runtime_pin=RUNTIME_PIN,
            expected_workload=workload,
            expected_acceptance=acceptance,
            expected_routes=routes,
            **_comparison_kwargs(),
        )

    reused = deepcopy(mtp_evidence)
    reused["report_file_sha256"] = no_evidence["report_file_sha256"]
    reused["evidence_sha256"] = perf.canonical_sha256({
        key: value for key, value in reused.items() if key != "evidence_sha256"
    })
    with pytest.raises(perf.DSparkMatchedPerformanceError, match="reuse one report"):
        perf.build_matched_result(
            no_mtp_evidence=no_evidence,
            mtp_evidence=reused,
            target_model_sha=TARGET_SHA,
            draft_model_sha=DRAFT_SHA,
            runtime_pin=RUNTIME_PIN,
            expected_workload=workload,
            expected_acceptance=acceptance,
            expected_routes=routes,
            **_comparison_kwargs(),
        )


def test_report_rejects_counter_contamination_and_reused_report(
    tmp_path, monkeypatch
):
    workload = _workload()
    acceptance = _acceptance(workload)
    routes = _routes()
    report = _report(perf.NO_MTP_ARM, workload, acceptance, routes)
    report["counters"]["after"]["generation_tokens"] += 1
    report["counters"]["delta"]["generation_tokens"] += 1
    report["report_sha256"] = perf.canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    path = _write_report(tmp_path, perf.NO_MTP_ARM, report)
    monkeypatch.setattr(
        perf,
        "validate_dspark_serve_manifest",
        lambda manifest, **_kwargs: manifest["serve_fingerprint"],
    )
    with pytest.raises(perf.DSparkMatchedPerformanceError, match="contaminated"):
        perf.validate_arm_report(
            report,
            report_path=path,
            arm=perf.NO_MTP_ARM,
            expected_target_sha=TARGET_SHA,
            expected_draft_sha=DRAFT_SHA,
            expected_workload=workload,
            expected_acceptance=acceptance,
            expected_routes=routes,
            expected_speculative_config=None,
            expected_launch_options={},
            expected_launch_switches=set(),
            expected_no_mtp_graph_capture=report["graph_capture"],
            requires_moe_marlin=True,
        )


def test_report_accepts_continuous_sampling_through_post_manifest(
    tmp_path, monkeypatch
):
    workload = _workload()
    acceptance = _acceptance(workload)
    routes = _routes()
    report = _report(perf.NO_MTP_ARM, workload, acceptance, routes)
    extra_post = deepcopy(report["memory"]["samples"][-1])
    extra_post.update({
        "sequence": len(report["memory"]["samples"]),
        "observed_at": "2026-08-13T00:00:08.500000Z",
    })
    report["memory"]["samples"].append(extra_post)
    report["report_sha256"] = perf.canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    path = _write_report(tmp_path, perf.NO_MTP_ARM, report)
    monkeypatch.setattr(
        perf,
        "validate_dspark_serve_manifest",
        lambda manifest, **_kwargs: manifest["serve_fingerprint"],
    )
    evidence = perf.validate_arm_report(
        report,
        report_path=path,
        arm=perf.NO_MTP_ARM,
        expected_target_sha=TARGET_SHA,
        expected_draft_sha=DRAFT_SHA,
        expected_workload=workload,
        expected_acceptance=acceptance,
        expected_routes=routes,
        expected_speculative_config=None,
        expected_launch_options={"--compilation-config": "no-mtp-graph"},
        expected_launch_switches=set(),
        expected_no_mtp_graph_capture=report["graph_capture"],
        requires_moe_marlin=True,
    )
    assert evidence["memory"]["phase_summary"]["post"]["count"] == 2


def test_full_report_rejects_file_swap_mtp_response_drift_and_insufficient_kv(
    tmp_path, monkeypatch
):
    workload = _workload()
    acceptance = _acceptance(workload)
    routes = _routes()
    monkeypatch.setattr(
        perf,
        "validate_dspark_serve_manifest",
        lambda manifest, **_kwargs: manifest["serve_fingerprint"],
    )
    common = {
        "expected_target_sha": TARGET_SHA,
        "expected_draft_sha": DRAFT_SHA,
        "expected_workload": workload,
        "expected_acceptance": acceptance,
        "expected_routes": routes,
        "requires_moe_marlin": True,
    }

    no_report = _report(perf.NO_MTP_ARM, workload, acceptance, routes)
    no_path = _write_report(tmp_path, perf.NO_MTP_ARM, no_report)
    swapped = deepcopy(no_report)
    swapped["served_model"] = "changed-after-file-write"
    swapped["report_sha256"] = perf.canonical_sha256({
        key: value for key, value in swapped.items() if key != "report_sha256"
    })
    with pytest.raises(perf.DSparkMatchedPerformanceError, match="JSON file"):
        perf.validate_arm_report(
            swapped,
            report_path=no_path,
            arm=perf.NO_MTP_ARM,
            expected_speculative_config=None,
            expected_launch_options={"--compilation-config": "no-mtp-graph"},
            expected_launch_switches=set(),
            expected_no_mtp_graph_capture=no_report["graph_capture"],
            **common,
        )

    insufficient = deepcopy(no_report)
    insufficient["memory"]["kv_cache"]["capacity_tokens"] -= 1
    insufficient["report_sha256"] = perf.canonical_sha256({
        key: value for key, value in insufficient.items() if key != "report_sha256"
    })
    insufficient_path = tmp_path / "insufficient-kv.json"
    insufficient_path.write_text(json.dumps(insufficient, sort_keys=True) + "\n")
    with pytest.raises(perf.DSparkMatchedPerformanceError, match="256K"):
        perf.validate_arm_report(
            insufficient,
            report_path=insufficient_path,
            arm=perf.NO_MTP_ARM,
            expected_speculative_config=None,
            expected_launch_options={"--compilation-config": "no-mtp-graph"},
            expected_launch_switches=set(),
            expected_no_mtp_graph_capture=insufficient["graph_capture"],
            **common,
        )

    low_headroom = deepcopy(no_report)
    low_headroom["memory"]["samples"][1]["mem_available_bytes"] = 7 * perf.GIB
    low_headroom["memory"].update({
        "ready_mem_available_bytes": 7 * perf.GIB,
        "startup_min_mem_available_bytes": 7 * perf.GIB,
        "minimum_mem_available_bytes": 7 * perf.GIB,
        "model_residency_bytes": 113 * perf.GIB,
        "startup_transient_bytes": 0,
        "measured_transient_bytes": 0,
    })
    low_headroom["report_sha256"] = perf.canonical_sha256({
        key: value for key, value in low_headroom.items() if key != "report_sha256"
    })
    low_path = tmp_path / "low-headroom.json"
    low_path.write_text(json.dumps(low_headroom, sort_keys=True) + "\n")
    with pytest.raises(perf.DSparkMatchedPerformanceError, match="110/8/4"):
        perf.validate_arm_report(
            low_headroom,
            report_path=low_path,
            arm=perf.NO_MTP_ARM,
            expected_speculative_config=None,
            expected_launch_options={"--compilation-config": "no-mtp-graph"},
            expected_launch_switches=set(),
            expected_no_mtp_graph_capture=low_headroom["graph_capture"],
            **common,
        )

    mtp_report = _report(perf.MTP_ARM, workload, acceptance, routes)
    mtp_report["mtp"]["acceptance"] = acceptance
    mtp_report["responses"][0]["content_sha256"] = "9" * 64
    mtp_report["warmup_responses"][0]["content_sha256"] = "9" * 64
    mtp_report["report_sha256"] = perf.canonical_sha256({
        key: value for key, value in mtp_report.items() if key != "report_sha256"
    })
    mtp_path = tmp_path / "mtp-response-drift.json"
    mtp_path.write_text(json.dumps(mtp_report, sort_keys=True) + "\n")
    with pytest.raises(perf.DSparkMatchedPerformanceError, match="exact acceptance"):
        perf.validate_arm_report(
            mtp_report,
            report_path=mtp_path,
            arm=perf.MTP_ARM,
            expected_speculative_config={"method": "dspark"},
            expected_launch_options={
                "--compilation-config": "mtp-graph",
                "--speculative-config": "dspark-k5",
            },
            expected_launch_switches=set(),
            expected_mtp_pre_manifest=mtp_report["pre_manifest"],
            expected_mtp_post_manifest=mtp_report["post_manifest"],
            expected_mtp_graph_capture=mtp_report["graph_capture"],
            **common,
        )
