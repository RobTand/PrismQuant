from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from prismaquant.gridbook_serving_runtime_pin import (
    GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
)

import prismaquant.validate_dspark_target_draft as dsv


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _acceptance() -> dict:
    before = {key: 100.0 for key in dsv._SNAPSHOT_KEYS}
    positions = [280.0, 180.0, 100.0, 50.0, 30.0]
    delta = {
        "generation_tokens": 1024.0,
        "drafts": 400.0,
        "draft_tokens": 2000.0,
        "accepted_tokens": sum(positions),
        **{
            f"accepted_position_{index}": value
            for index, value in enumerate(positions)
        },
    }
    after = {key: before[key] + delta[key] for key in dsv._SNAPSHOT_KEYS}
    started = datetime(2026, 8, 13, tzinfo=timezone.utc)
    wall = 64.0
    model = "dsv4-flash-gridbook-" + "1" * 32
    return {
        "schema": dsv.DSPARK_ACCEPTANCE_SCHEMA,
        "started_at": _utc(started),
        "finished_at": _utc(started + timedelta(seconds=wall)),
        "base_url": "http://127.0.0.1:8000",
        "served_model": model,
        "workload": {
            "schema": dsv.DSPARK_WORKLOAD_SCHEMA,
            "prompt_count": len(dsv.DSPARK_PROMPTS),
            "prompt_sha256": list(dsv.DSPARK_PROMPT_SHA256),
            "prompt_corpus_sha256": dsv.DSPARK_PROMPT_CORPUS_SHA256,
            "max_tokens_per_prompt": dsv.DSPARK_MAX_TOKENS,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
        },
        "before": before,
        "after": after,
        "delta": delta,
        "wall_seconds": wall,
        "served_output_tokens_per_second": 1024.0 / wall,
        "responses": [
            {
                "index": index,
                "prompt_sha256": dsv.DSPARK_PROMPT_SHA256[index],
                "content_sha256": hashlib.sha256(
                    f"response-{index}".encode()
                ).hexdigest(),
                "finish_reason": "length",
                "prompt_tokens": 10 + index,
                "completion_tokens": 128,
                "request_seconds": 8.0,
            }
            for index in range(8)
        ],
        "acceptance": {
            "drafts": 400.0,
            "draft_tokens": 2000.0,
            "accepted_tokens": sum(positions),
            "accepted_tokens_per_position": positions,
            "aggregate_rate": sum(positions) / 2000.0,
            "accepted_tokens_per_cycle": sum(positions) / 400.0,
            "mean_acceptance_length": 1.0 + sum(positions) / 400.0,
            "per_position_rates": [value / 400.0 for value in positions],
            "position_zero_gate_minimum": 0.60,
            "position_zero_gate_passed": True,
        },
    }


def _runtime_pin() -> dict:
    return {
        "schema": "prismaquant.gridbook_serving_runtime_pin.v1",
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "a" * 40,
        "version": GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
        "version_is_release": True,
        "wheel_sha256": "b" * 64,
        "runtime_contract_schema": "gridbook.runtime-contract.v4",
        "required_abi_features": {
            "routed_moe_per_role_codebook_lut": 1,
            "source_fp8_block128_w8a16": 1,
            "dspark_construction_physical_bridge": 1,
        },
    }


def _binding(model_sha: str, launch_model: str) -> dict:
    return {
        "schema": "prismaquant.served_artifact_binding/1",
        "resolved_path": launch_model,
        "launch_model": launch_model,
        "model_sha": model_sha,
        "artifact_inventory_sha256": "c" * 64,
        "artifact_bytes": 1024,
    }


def _pair(monkeypatch) -> dict:
    acceptance = _acceptance()
    monkeypatch.setattr(dsv, "_gridbook_runtime_pin", _runtime_pin)
    monkeypatch.setattr(
        dsv, "validate_serving_profile_receipt", lambda payload, **_kwargs: dict(payload)
    )
    monkeypatch.setattr(
        dsv, "validate_runtime_evidence", lambda payload, **_kwargs: dict(payload)
    )
    monkeypatch.setattr(
        dsv, "validate_route_census", lambda payload, **_kwargs: dict(payload)
    )
    monkeypatch.setattr(
        dsv,
        "validate_baseline_comparison_evidence",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        dsv, "validate_matched_result", lambda payload, **_kwargs: dict(payload)
    )
    target_sha = "d" * 64
    draft_sha = "e" * 64
    source = {
        "schema": "prismaquant.streamed_model.identity.v1",
        "content_sha256": "f" * 64,
        "resolved_commit": None,
        "checkpoint_shards": 48,
        "checkpoint_tensors": 72317,
    }
    profile = {"receipt_sha256": "0" * 64}
    runtime = {
        "profile_receipt": profile,
        "evidence_sha256": "8" * 64,
    }
    target_binding = _binding(target_sha, "/model")
    draft_binding = _binding(draft_sha, "/draft")
    result = {
        "schema": dsv.DSPARK_PAIR_RESULT_SCHEMA,
        "target_model_sha": target_sha,
        "draft_model_sha": draft_sha,
        "source_model_identity": source,
        "target_decode_contract_sha256": "1" * 64,
        "draft_decode_contract_sha256": "2" * 64,
        "draft_render_attestation": {
            "schema": "prismaquant.dspark_cb_render_recipe.v1",
            "source_model_identity": source,
            "recipe_sha256": "3" * 64,
            "source_weights_sha256": "4" * 64,
            "source_weights_entries": 2325,
            "attestation_sha256": "5" * 64,
        },
        "target_artifact_binding": target_binding,
        "draft_artifact_binding": draft_binding,
        "runtime_pin": _runtime_pin(),
        "dspark_serving_profile": profile,
        "dspark_runtime_evidence": runtime,
        "serving_stack": {
            "image": dsv.DSV4_SPARK_VLLM_IMAGE,
            "vllm_version": dsv.DSV4_SPARK_VLLM_VERSION,
            "speculative_config": dict(dsv.DSPARK_SPECULATIVE_CONFIG),
            "launch_options": dsv._expected_launch_options(
                acceptance["served_model"]
            ),
            "launch_switches": sorted(dsv.DSPARK_LAUNCH_SWITCHES),
        },
        "serve_session_id": "6" * 64,
        "pre_serve_fingerprint": "7" * 64,
        "post_serve_fingerprint": "7" * 64,
        "pre_manifest_sha256": "9" * 64,
        "post_manifest_sha256": "a" * 64,
        "graph_capture": {
            "serve_log_sha256": "b" * 64,
            "capture_marker": "Graph capturing finished in 1 secs, took -0.10 GiB",
            "capture_sizes": [5, 6],
        },
        "route_census": {
            "serve_log_sha256": "b" * 64,
        },
        "acceptance_suite": acceptance,
        "acceptance_suite_sha256": "c" * 64,
        "baseline_comparison": {
            "local_model": {
                "target_model_sha256": target_sha,
                "draft_model_sha256": draft_sha,
            },
        },
        "baseline_comparison_file_sha256": "0" * 64,
        "matched_performance": {
            "tool": {"git_commit": "f" * 40},
            "mtp": {
                "stable_serve_fingerprint": "7" * 64,
                "target_artifact_binding": target_binding,
                "draft_artifact_binding": draft_binding,
            },
            "common_manifest_identity": {
                "dspark_serving_profile_sha256": profile["receipt_sha256"],
                "dspark_runtime_evidence_sha256": runtime["evidence_sha256"],
            },
        },
    }
    result["receipt_sha256"] = dsv._canonical_sha256(result)
    return result


def test_acceptance_replays_fixed_token_and_metric_arithmetic():
    payload = _acceptance()
    assert dsv.validate_acceptance_suite(
        payload, expected_served_model=payload["served_model"]
    )["acceptance"]["position_zero_gate_passed"] is True


def test_acceptance_rejects_contamination_and_weak_first_position():
    contaminated = _acceptance()
    contaminated["after"]["generation_tokens"] += 1
    contaminated["delta"]["generation_tokens"] += 1
    with pytest.raises(dsv.DSparkValidationError, match="contaminated"):
        dsv.validate_acceptance_suite(
            contaminated,
            expected_served_model=contaminated["served_model"],
        )

    weak = _acceptance()
    weak_positions = [200.0, 180.0, 100.0, 50.0, 30.0]
    weak_accepted = sum(weak_positions)
    weak["delta"]["accepted_tokens"] = weak_accepted
    weak["after"]["accepted_tokens"] = (
        weak["before"]["accepted_tokens"] + weak_accepted
    )
    for index, value in enumerate(weak_positions):
        key = f"accepted_position_{index}"
        weak["delta"][key] = value
        weak["after"][key] = weak["before"][key] + value
    weak["acceptance"].update({
        "accepted_tokens": weak_accepted,
        "accepted_tokens_per_position": weak_positions,
        "aggregate_rate": weak_accepted / 2000.0,
        "accepted_tokens_per_cycle": weak_accepted / 400.0,
        "mean_acceptance_length": 1.0 + weak_accepted / 400.0,
        "per_position_rates": [value / 400.0 for value in weak_positions],
        "position_zero_gate_passed": False,
    })
    with pytest.raises(dsv.DSparkValidationError, match="summary/gate"):
        dsv.validate_acceptance_suite(
            weak, expected_served_model=weak["served_model"]
        )


def test_pair_receipt_and_both_shipcard_roles_are_fail_closed(monkeypatch):
    result = _pair(monkeypatch)
    assert dsv.validate_pair_result(result)["receipt_sha256"] == result[
        "receipt_sha256"
    ]
    for role in ("target", "draft"):
        peer = "draft" if role == "target" else "target"
        record = {
            "slot": dsv.DSPARK_SHIPCARD_SLOT,
            "tool": dsv.DSPARK_SHIPCARD_TOOL,
            "filled_at": "2026-08-13T00:02:00Z",
            "passed": True,
            "model_sha": result[f"{role}_model_sha"],
            "spec_decode_detected": True,
            "serve_fingerprint": result["post_serve_fingerprint"],
            "git_commit": "f" * 40,
            "detail": "passed",
            "metrics": result,
            "artifact_role": role,
            "peer_model_sha": result[f"{peer}_model_sha"],
        }
        assert dsv.validate_dspark_shipcard_record(
            dsv.DSPARK_SHIPCARD_SLOT, record
        ) == []
        wrong_collector = deepcopy(record)
        wrong_collector["git_commit"] = "0" * 40
        assert dsv.validate_dspark_shipcard_record(
            dsv.DSPARK_SHIPCARD_SLOT, wrong_collector
        )

    stale = deepcopy(result)
    stale["draft_artifact_binding"]["model_sha"] = "0" * 64
    with pytest.raises(dsv.DSparkValidationError):
        dsv.validate_pair_result(stale)

    changed_stack = deepcopy(result)
    changed_stack["post_serve_fingerprint"] = "8" * 64
    changed_stack["receipt_sha256"] = dsv._canonical_sha256({
        key: value
        for key, value in changed_stack.items()
        if key != "receipt_sha256"
    })
    with pytest.raises(
        dsv.DSparkValidationError, match="stable serve fingerprints differ"
    ):
        dsv.validate_pair_result(changed_stack)


def test_graph_capture_requires_dspark_five_token_shapes(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text(
        "SpeculativeConfig(method='dspark', model='/draft', num_spec_tokens=5)\n"
        "'cudagraph_capture_sizes': [5, 6], 'max_cudagraph_capture_size': 6\n"
        "Graph capturing finished in 1 secs, took -0.10 GiB\n"
    )
    assert dsv.validate_dspark_graph_capture_log(log)["capture_sizes"] == [5, 6]
    log.write_text(log.read_text() + "Skipping CUDA graph capture\n")
    with pytest.raises(dsv.DSparkValidationError, match="does not prove"):
        dsv.validate_dspark_graph_capture_log(log)


def test_matched_no_mtp_graph_requires_compiled_single_token_shape(tmp_path):
    log = tmp_path / "no-mtp-serve.log"
    log.write_text(
        "'cudagraph_capture_sizes': [1], 'max_cudagraph_capture_size': 1\n"
        "Graph capturing finished in 1 secs, took -0.05 GiB\n"
    )
    assert dsv.validate_no_mtp_graph_capture_log(log)["capture_sizes"] == [1]
    log.write_text(
        log.read_text()
        + "SpeculativeConfig(method='dspark', model='/draft', num_spec_tokens=5)\n"
    )
    with pytest.raises(dsv.DSparkValidationError, match="non-speculative"):
        dsv.validate_no_mtp_graph_capture_log(log)
