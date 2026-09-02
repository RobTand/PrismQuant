"""Adversarial replay for the single-Spark DSv4 Gridbook gold receipts."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import pytest

from prismaquant.gridbook_environment import (
    CANONICAL_GOLD_SET_ENVIRONMENT,
)
from prismaquant.gridbook_runtime_pin import load_gridbook_runtime_pin
from tools.full_kl_teacher_payload import PROMPT_TOP_K
from prismaquant.shipcard import (
    DSV4_TOKENIZER_IDENTITY_SHA256,
    DSV4_WIKITEXT_CORPUS_SHA256,
    DSV4_WIKITEXT_DATASET_FINGERPRINT,
    DSV4_WIKITEXT_SELECTED_TOKEN_IDS_SHA256,
    DSV4_WIKITEXT_TOTAL_TOKENS,
    SHIPCARD_RESERVED_BYTES,
    _verify_dsv4_gridbook_gold_contract,
    build_shipcard,
    build_weight_content_manifest,
    load_shipcard,
    make_record,
    reattest_weight_stats,
    verify,
    write_shipcard,
)
from prismaquant.shipcard_cli import CARRIED_METRIC_KEYS
from prismaquant.shipcard_cli import main as shipcard_cli
from prismaquant.validate_cb_endpoint import (
    DSV4_SPARK_GPU_NAME,
    DSV4_SPARK_VLLM_IMAGE,
    DSV4_SPARK_VLLM_VERSION,
)
from tools.dsv4_gridbook_contract import exact_llm_contract
from tools.serve_fingerprint import (
    _GOLD_PRODUCER_COMMON_FILES,
    _GOLD_PRODUCER_TOOL_FILES,
    SERVER_ENV_ALLOWLIST,
    artifact_binding,
    elide_argv_paths,
    fingerprint,
    normalize_performance_argv,
    performance_stack_fingerprint,
    process_identity_sha256,
    serve_session_fingerprint,
)


_SOURCE = {
    "schema": "prismaquant.streamed_model.identity.v1",
    "content_sha256": "e" * 64,
    "resolved_commit": None,
    "checkpoint_shards": 48,
    "checkpoint_tensors": 72_317,
}
_GOLD_ENV = {
    **dict(CANONICAL_GOLD_SET_ENVIRONMENT),
    "PYTHONSAFEPATH": "1",
}


@pytest.fixture(autouse=True)
def _resolved_release_pin_for_gold_replay(monkeypatch):
    """Unit-test gold replay mechanics against the exact released pin."""

    import prismaquant.validate_cb_endpoint as endpoint

    released = load_gridbook_runtime_pin()
    monkeypatch.setattr(
        sys.modules[__name__], "load_gridbook_runtime_pin", lambda: released
    )
    monkeypatch.setattr(endpoint, "_gridbook_runtime_pin", lambda: {
        "schema": released.schema,
        "repository": released.repository,
        "commit": released.commit,
        "version": released.version,
        "version_is_release": released.version_is_release,
        "runtime_contract_schema": released.runtime_contract_schema,
        "required_abi_features": dict(released.required_abi_features),
    })


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finalize_inventory(root: Path, quant_config: dict) -> None:
    quant_path = root / "quant_config.json"
    for _ in range(20):
        _write_json(quant_path, quant_config)
        ledger = {
            path.relative_to(root).as_posix(): int(path.stat().st_size)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        inventory = {
            "schema": "prismaquant.cb_export_artifact_inventory.v1",
            "scope": "all_regular_files_recursive",
            "file_bytes": ledger,
            "export_directory_bytes": sum(ledger.values()),
            "whole_artifact_budget_bytes": 112_690_000_000,
        }
        if quant_config["provenance"].get("artifact_inventory") == inventory:
            return
        quant_config["provenance"]["artifact_inventory"] = inventory
    raise AssertionError("test artifact inventory did not converge")


def _artifact(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "config.json").write_text(
        '{"model_type":"deepseek_v4","num_hidden_layers":43}\n',
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"final weights")
    (root / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"TestTokenizer"}\n', encoding="utf-8"
    )
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "layout_version": 2,
        "config_groups": {
            "group_0": {
                "format": "NVFP4_CB_K12",
                "targets": ["model.layers.0.mlp.experts"],
                "scheme": {"type": "group"},
            },
        },
        "source_passthrough": {
            "version": 1,
            "units": {
                "model.layers.0.self_attn.q_proj": (
                    "fp8_e4m3_ue8m0_block128"
                ),
            },
        },
        "provenance": {
            "source_model_identity": dict(_SOURCE),
            "weight_content_manifest": build_weight_content_manifest(root),
            "artifact_inventory": {
                "schema": "prismaquant.cb_export_artifact_inventory.v1",
                "scope": "pending_final_write",
            },
        },
    }
    _write_json(root / "quant_config.json", quant_config)
    card = build_shipcard(root, build={"quant_method": "gridbook"})
    write_shipcard(root / "shipcard.json", card)
    _finalize_inventory(root, quant_config)
    return root, card


def _process(pid: int, argv: list[str], *, boot_id: str) -> dict:
    row = {
        "pid": pid,
        "argv": argv,
        "cmdline": " ".join(argv),
        "start_time_ticks": 100_000 + pid,
        "pid_namespace": "pid:[4026533000]",
        "executable": "/usr/local/bin/python3.12",
    }
    row["identity_sha256"] = process_identity_sha256(row, boot_id=boot_id)
    return row


def _teacher_evidence() -> dict:
    calibration = {
        "schema": "prismaquant.wikitext_gold_calibration/1",
        "dataset": {
            "name": "wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "train",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "fingerprint": "immutable-dataset-fingerprint",
            "corpus_sha256": "1" * 64,
        },
        "corpus_construction": {
            "row_filter": "include iff bool(text.strip()); preserve text verbatim",
            "join_separator": "\n\n",
            "normalization": "none",
        },
        "tokenizer": {
            "identity_sha256": "2" * 64,
            "trust_remote_code": True,
            "add_special_tokens": False,
        },
        "window_seed": 42,
        "sampler": (
            "python.random.Random(seed).sample(range(max_start), n_samples)/v1"
        ),
        "n_samples": 8,
        "seqlen": 512,
        "starts": list(range(10, 18)),
        "total_tokens": 100_000,
        "calib_ids_sha256": "3" * 64,
        "scoring": {
            "positions": "all",
            "prompt_top_k": PROMPT_TOP_K,
            "logprob_dtype": "float32",
            "tail_bucket": True,
        },
    }
    return {
        "schema": "prismaquant.full_kl_teacher_evidence/1",
        "payload_sha256": "4" * 64,
        "payload_bytes": 123_456,
        "payload_semantic_sha256": "5" * 64,
        "meta_sha256": "6" * 64,
        "source_model": dict(_SOURCE),
        "source_model_identity_sha256": "7" * 64,
        "calibration_contract": calibration,
        "calibration_contract_sha256": _canonical_sha(calibration),
        "topk_coverage_mean": 0.97,
        "topk_coverage_min": 0.91,
        "topk_coverage_policy": {
            "schema": "prismaquant.topk_tail_coverage_policy/1",
            "top_k": PROMPT_TOP_K,
            "minimum_probability_mass_per_position": 0.90,
            "maximum_probability_mass": 1.0,
            "probability_mass_absolute_tolerance": 1e-6,
            "maximum_declared_tail_mass_per_position": 1.0 - 0.90,
            "tail_bucket": True,
        },
    }


def _ppl_calibration(_root: Path) -> dict:
    return {
        "schema": "prismaquant.wikitext_ppl_calibration/1",
        "dataset": {
            "name": "wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "fingerprint": DSV4_WIKITEXT_DATASET_FINGERPRINT,
            "corpus_sha256": DSV4_WIKITEXT_CORPUS_SHA256,
        },
        "corpus_construction": {
            "row_filter": "include iff bool(text.strip()); preserve text verbatim",
            "join_separator": "\n\n",
            "normalization": "none",
        },
        "tokenizer": {
            "identity_sha256": DSV4_TOKENIZER_IDENTITY_SHA256,
            "trust_remote_code": True,
            "add_special_tokens": False,
        },
        "token_selection": {
            "strategy": "contiguous_prefix_after_full_corpus_tokenization/v1",
            "n_tokens_requested": 8192,
            "n_tokens_available": DSV4_WIKITEXT_TOTAL_TOKENS,
            "selected_token_count": 8192,
            "token_ids_sha256": DSV4_WIKITEXT_SELECTED_TOKEN_IDS_SHA256,
            "digest_encoding": "canonical_json_integer_array/v1",
        },
        "scoring": {
            "chunking": "nonoverlapping_contiguous/v1",
            "seqlen": 512,
            "chunk_starts": list(range(0, 8192, 512)),
            "chunk_token_counts": [512] * 16,
            "positions": "within_each_chunk_positions_1_through_N_minus_1",
            "n_tokens_scored": 8176,
            "prompt_logprobs": 1,
            "temperature": 0.0,
            "max_tokens": 1,
            "detokenize": False,
        },
    }


def _producer_identity(tool: str) -> dict:
    files = {
        name: {"bytes": 123, "sha256": "9" * 64}
        for name in sorted(set(
            _GOLD_PRODUCER_COMMON_FILES + _GOLD_PRODUCER_TOOL_FILES[tool]
        ))
    }
    return {
        "schema": "prismaquant.gold_producer_identity/1",
        "measurement_tool": tool,
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "git_dirty": False,
        "source_files": files,
        "source_files_sha256": _canonical_sha(files),
    }


def _gridbook_distribution(pin) -> dict:
    source_files = {
        name: {"bytes": 123, "sha256": "c" * 64}
        for name in (
            "gridbook/__init__.py",
            "gridbook/cuda_ext.py",
            "gridbook/plugin.py",
            "gridbook/runtime_contract.json",
            "gridbook/source_passthrough.py",
            "gridbook/fp8_source_w8a16.py",
            "gridbook/csrc/cb_gemv.cu",
            "gridbook/csrc/fp8_source_w8a16.cu",
            "gridbook/csrc/mxfp8_dense_gemm.cu",
        )
    }
    package_root = "/usr/local/lib/python3.12/site-packages/gridbook"
    import_origin = {
        "schema": "prismaquant.gridbook_import_origin/1",
        "module_name": "gridbook",
        "imported_version": pin.version,
        "distribution_package_root": package_root,
        "module_file": f"{package_root}/__init__.py",
        "module_search_locations": [package_root],
    }
    import_origin["identity_sha256"] = _canonical_sha(import_origin)
    return {
        "schema": "prismaquant.installed_gridbook_distribution/2",
        "name": "gridbook",
        "repository": pin.repository,
        "version": pin.version,
        "direct_url": {
            "url": f"file:///tmp/gridbook-runtime-{pin.commit[:12]}",
            "vcs_info": {
                "vcs": "git",
                "requested_revision": pin.commit,
                "commit_id": pin.commit,
            },
        },
        "direct_url_path": f"gridbook-{pin.version}.dist-info/direct_url.json",
        "direct_url_identity": {"bytes": 123, "sha256": "d" * 64},
        "metadata_path": f"gridbook-{pin.version}.dist-info/METADATA",
        "metadata_identity": {"bytes": 123, "sha256": "e" * 64},
        "record_path": f"gridbook-{pin.version}.dist-info/RECORD",
        "record_identity": {"bytes": 123, "sha256": "f" * 64},
        "source_files": source_files,
        "source_files_sha256": _canonical_sha(source_files),
        "import_origin": import_origin,
    }


def _gold_record(root: Path, *, slot: str = "gold.kl") -> tuple[dict, dict]:
    kwargs, contract = exact_llm_contract(root)
    contract_sha = _canonical_sha(contract)
    pin = load_gridbook_runtime_pin()
    launch = [
        "/usr/local/bin/python3.12",
        "/repo/tools/measure_vllm_full_kl.py",
        "--mode", "student",
        "--model", str(root.resolve()),
        "--teacher-payload", "/evidence/teacher.pt",
        "--teacher-meta", "/evidence/teacher.json",
        "--dsv4-gridbook-contract",
    ]
    boot_id = "11111111-2222-3333-4444-555555555555"
    parent_pid, engine_pid = 1001, 1002
    processes = [
        _process(parent_pid, launch, boot_id=boot_id),
        _process(engine_pid, ["VLLM::EngineCore"], boot_id=boot_id),
    ]
    environment_rows = [{
        "pid": pid,
        "values": dict(_GOLD_ENV),
        "sha256": _canonical_sha(_GOLD_ENV),
    } for pid in (parent_pid, engine_pid)]
    manifest = {
        "schema": "prismaquant.serve_manifest/1",
        "created": "2026-08-12T12:00:00Z",
        "attestation_phase": "snapshot",
        "source": "in_process",
        "hostname": "one-spark",
        "host_identity": {
            "hostname": "one-spark",
            "boot_id": boot_id,
            "machine_id_sha256": "8" * 64,
            "pid_namespace": "pid:[4026533000]",
        },
        "image": DSV4_SPARK_VLLM_IMAGE,
        "model": str(root.resolve()),
        "served_model_name": None,
        "launch_argv": launch,
        "launch_flags": elide_argv_paths(launch),
        "normalized_performance_argv": normalize_performance_argv(launch),
        "enforce_eager": False,
        "quantization": None,
        "kv_cache_dtype": None,
        "speculative_config": None,
        "package_versions": {
            "gridbook": pin.version,
            "vllm": DSV4_SPARK_VLLM_VERSION,
        },
        "gridbook_runtime_pin": {
            "commit": pin.commit,
            "version": pin.version,
        },
        "gridbook_distribution": _gridbook_distribution(pin),
        "resident_extensions": sorted([
            "pq_fp8_source_w8a16_deadbeef.so",
            "pq_cb_bf16_grouped_deadbeef.so",
            "prismaquant_cb_ext.so",
            "prismaquant_cb_v2_ext.so",
        ]),
        "residency_readable": True,
        "processes": processes,
        "server_process_environment": {
            "schema": "prismaquant.server_process_environment/1",
            "allowlist": sorted(SERVER_ENV_ALLOWLIST),
            "readable_pids": [parent_pid, engine_pid],
            "unreadable_pids": [],
            "consistent": True,
            "values": dict(_GOLD_ENV),
            "processes": environment_rows,
        },
        "pq_env": dict(_GOLD_ENV),
        "listener_census": {
            "schema": "prismaquant.server_tcp_listeners/1",
            "tables_readable": True,
            "unreadable_pids": [],
            "listeners": [],
        },
        "listener_binding": None,
        "gpu_name": DSV4_SPARK_GPU_NAME,
        "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
        "driver_version": "release-driver",
        "gpu_count": 1,
        "artifact_binding": artifact_binding(root, launch_model=root),
        "measurement_tool": "measure_vllm_full_kl",
        "producer_identity": _producer_identity("measure_vllm_full_kl"),
        "effective_llm_kwargs": kwargs,
        "dsv4_gridbook_contract_sha256": contract_sha,
        "measurement_parent_pid": parent_pid,
        "engine_descendant_pids": [engine_pid],
    }
    manifest["serve_session_id"] = serve_session_fingerprint(manifest)
    manifest["performance_stack_fingerprint"] = performance_stack_fingerprint(
        manifest
    )
    manifest["serve_fingerprint"] = fingerprint(manifest)
    metrics = {
        "mode": "student",
        "score_positions": "all",
        "prompt_top_k": PROMPT_TOP_K,
        "model": str(root),
        "quantization": "gridbook",
        "n_samples": 8,
        "seqlen": 512,
        "vocab_size": 129_280,
        "n_positions": 4088,
        "kl_mean": 0.2,
        "kl_p99": 0.4,
        "kl_max": 0.5,
        "kl_confident_mean": 0.1,
        "n_confident": 100,
        "teacher_evidence": _teacher_evidence(),
        "dsv4_gridbook_contract": contract,
        "dsv4_gridbook_contract_sha256": contract_sha,
        "serve_manifest": manifest,
    }
    record = make_record(
        slot=slot,
        tool="record:gold.json",
        passed=True,
        model_sha=manifest["artifact_binding"]["model_sha"],
        metrics=metrics,
        spec_decode_detected=False,
        serve_fingerprint=manifest["serve_fingerprint"],
        git_commit="a" * 40,
    )
    return record, metrics


def _resign(record: dict) -> None:
    manifest = record["metrics"]["serve_manifest"]
    manifest["serve_session_id"] = serve_session_fingerprint(manifest)
    manifest["performance_stack_fingerprint"] = performance_stack_fingerprint(
        manifest
    )
    manifest["serve_fingerprint"] = fingerprint(manifest)
    record["serve_fingerprint"] = manifest["serve_fingerprint"]


def _ppl_record(root: Path) -> tuple[dict, dict]:
    record, metrics = _gold_record(root, slot="gold.ppl")
    calibration = _ppl_calibration(root)
    manifest = metrics["serve_manifest"]
    manifest["measurement_tool"] = "measure_vllm_wikitext_ppl"
    manifest["producer_identity"] = _producer_identity(
        "measure_vllm_wikitext_ppl"
    )
    manifest["launch_argv"] = [
        "/usr/local/bin/python3.12",
        "/repo/tools/measure_vllm_wikitext_ppl.py",
        "--model", str(root.resolve()),
        "--dsv4-gridbook-contract",
    ]
    manifest["launch_flags"] = elide_argv_paths(manifest["launch_argv"])
    manifest["normalized_performance_argv"] = normalize_performance_argv(
        manifest["launch_argv"]
    )
    per_chunk = [1.0 + index / 100.0 for index in range(16)]
    mean_nll = math.fsum(per_chunk) / len(per_chunk)
    metrics.update({
        "split": "test",
        "n_tokens_requested": 8192,
        "n_tokens_scored": 8176,
        "seqlen": 512,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "per_chunk_mean_nll": per_chunk,
        "max_chunk_mean_nll": max(per_chunk),
        "calibration_contract": calibration,
        "calibration_contract_sha256": _canonical_sha(calibration),
    })
    record["metrics"].update(metrics)
    metrics = record["metrics"]
    _resign(record)
    return record, metrics


def test_gold_replay_accepts_null_source_commit_and_exact_engine_evidence(
    tmp_path, monkeypatch,
):
    root, _card = _artifact(tmp_path)
    record, metrics = _gold_record(root)

    assert _verify_dsv4_gridbook_gold_contract(
        "gold.kl", record, metrics, model_dir=root
    ) == []


def test_gold_replay_reports_an_unavailable_release_pin(tmp_path, monkeypatch):
    import prismaquant.validate_cb_endpoint as endpoint

    root, _card = _artifact(tmp_path)
    record, metrics = _gold_record(root)

    def unavailable():
        raise endpoint.CBEndpointValidationError("staged release pin")

    monkeypatch.setattr(endpoint, "_gridbook_runtime_pin", unavailable)
    problems = _verify_dsv4_gridbook_gold_contract(
        "gold.kl", record, metrics, model_dir=root
    )
    assert any("release pin unavailable" in problem for problem in problems)


def test_ppl_calibration_is_pinned_and_artifact_tokenizer_bound(tmp_path):
    root, _card = _artifact(tmp_path)
    record, metrics = _ppl_record(root)
    calibration = metrics["calibration_contract"]

    problems = _verify_dsv4_gridbook_gold_contract(
        "gold.ppl", record, metrics, model_dir=None
    )
    assert not any("PPL calibration" in item for item in problems), problems
    assert not any("PPL tokenizer" in item for item in problems), problems
    assert not any("PPL token-prefix" in item for item in problems), problems
    assert not any("PPL scoring-window" in item for item in problems), problems

    # A receipt with the exact release tokenizer identity remains tied to the
    # files in the artifact; this deliberately synthetic tokenizer must fail.
    artifact_problems = _verify_dsv4_gridbook_gold_contract(
        "gold.ppl", record, metrics, model_dir=root
    )
    assert any(
        "PPL tokenizer identity differs from artifact" in item
        for item in artifact_problems
    )

    calibration["dataset"]["revision"] = "moving-main"
    metrics["calibration_contract_sha256"] = _canonical_sha(calibration)
    problems = _verify_dsv4_gridbook_gold_contract(
        "gold.ppl", record, metrics, model_dir=root
    )
    assert any("PPL calibration dataset identity" in item for item in problems)


def test_cli_carries_per_chunk_ppl_evidence():
    assert "per_chunk_mean_nll" in CARRIED_METRIC_KEYS


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda metrics: metrics.pop("mean_nll"), "PPL mean_nll"),
        (lambda metrics: metrics.__setitem__("mean_nll", math.nan),
         "PPL mean_nll"),
        (lambda metrics: metrics.__setitem__("mean_nll", -0.01),
         "PPL mean_nll"),
        (lambda metrics: metrics.pop("ppl"), "PPL ppl"),
        (lambda metrics: metrics.__setitem__("ppl", math.inf), "PPL ppl"),
        (lambda metrics: metrics.__setitem__("ppl", 0.99), "PPL ppl"),
        (lambda metrics: metrics.__setitem__(
            "per_chunk_mean_nll", metrics["per_chunk_mean_nll"][:-1]
        ), "exactly 16"),
        (lambda metrics: metrics["per_chunk_mean_nll"].__setitem__(
            4, math.nan
        ), "exactly 16"),
        (lambda metrics: metrics["per_chunk_mean_nll"].__setitem__(
            4, -0.1
        ), "exactly 16"),
        (lambda metrics: metrics.__setitem__("max_chunk_mean_nll", 99.0),
         "max_chunk_mean_nll arithmetic"),
        (lambda metrics: metrics.__setitem__(
            "mean_nll", metrics["mean_nll"] + 0.01
        ), "mean_nll differs from the per-chunk mean"),
        (lambda metrics: metrics.__setitem__(
            "ppl", metrics["ppl"] + 0.01
        ), "ppl differs from exp(mean_nll)"),
    ],
)
def test_ppl_arithmetic_evidence_is_adversarially_replayed(
    tmp_path, mutate, fragment,
):
    root, _card = _artifact(tmp_path)
    record, metrics = _ppl_record(root)
    mutate(metrics)

    problems = _verify_dsv4_gridbook_gold_contract(
        "gold.ppl", record, metrics, model_dir=None
    )
    assert any(fragment in problem for problem in problems), problems


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda contract: contract["dataset"].__setitem__(
            "fingerprint", "mutable-fingerprint"
        ), "PPL calibration dataset identity"),
        (lambda contract: contract["dataset"].__setitem__(
            "corpus_sha256", "1" * 64
        ), "PPL calibration dataset identity"),
        (lambda contract: contract["tokenizer"].__setitem__(
            "identity_sha256", "2" * 64
        ), "PPL tokenizer identity"),
        (lambda contract: contract["token_selection"].__setitem__(
            "n_tokens_available", 287_598
        ), "PPL token-prefix identity"),
        (lambda contract: contract["token_selection"].__setitem__(
            "token_ids_sha256", "3" * 64
        ), "PPL token-prefix identity"),
        (lambda contract: contract["scoring"].__setitem__(
            "temperature", 0.1
        ), "PPL scoring-window contract"),
        (lambda contract: contract["scoring"].__setitem__(
            "max_tokens", 2
        ), "PPL scoring-window contract"),
    ],
)
def test_ppl_value_identity_and_sampling_contract_are_exact(
    tmp_path, mutate, fragment,
):
    root, _card = _artifact(tmp_path)
    record, metrics = _ppl_record(root)
    mutate(metrics["calibration_contract"])
    metrics["calibration_contract_sha256"] = _canonical_sha(
        metrics["calibration_contract"]
    )

    problems = _verify_dsv4_gridbook_gold_contract(
        "gold.ppl", record, metrics, model_dir=None
    )
    assert any(fragment in problem for problem in problems), problems


def test_gold_receipt_survives_a_model_card_but_no_other_file(tmp_path):
    """Documenting an artifact must not invalidate its gold records.

    `compute_model_sha` excludes exactly `README.md` so a card can quote the
    gold numbers measured before it existed; the inventory replay must honor
    the same one-filename doctrine (the exporter's finalized inventory
    predates any card by construction).  Every OTHER added file remains a
    binding violation.
    """
    root, _card = _artifact(tmp_path)
    record, _metrics = _gold_record(root)
    kwargs = dict(model_dir=root, require_current_artifact_path=True)
    assert _verify_dsv4_gridbook_gold_contract(
        "gold.kl", record, record["metrics"], **kwargs
    ) == []

    (root / "README.md").write_text("# model card, added after the gates\n")
    assert _verify_dsv4_gridbook_gold_contract(
        "gold.kl", record, record["metrics"], **kwargs
    ) == []

    # The card figures share the doctrine (rendered from the attested
    # quant_config after the gates); any other name stays a violation.
    (root / "allocation-map.png").write_bytes(b"\x89PNG\r\n")
    (root / "byte-budget.png").write_bytes(b"\x89PNG\r\n")
    assert _verify_dsv4_gridbook_gold_contract(
        "gold.kl", record, record["metrics"], **kwargs
    ) == []

    (root / "extra.bin").write_bytes(b"\0")
    problems = _verify_dsv4_gridbook_gold_contract(
        "gold.kl", record, record["metrics"], **kwargs
    )
    assert any("differ from inventory" in p for p in problems)


def test_artifact_binding_tolerates_documentation_but_no_other_file(tmp_path):
    """`serve_fingerprint.artifact_binding` walks the SERVED dir against the
    finalized inventory at measurement-provenance time; a published artifact
    legitimately carries its README, card figures, and shipcard — all written
    after the exporter finalized the inventory. The walk honors the same
    exact-filename doctrine (and `shipcard.json`, the gate record itself),
    but only for names the inventory does NOT list: an inventoried file can
    never dodge its byte check by wearing a documentation name."""
    import pytest

    root, _card = _artifact(tmp_path)
    assert artifact_binding(root, launch_model=root)["model_sha"]

    (root / "README.md").write_text("# model card, added after the gates\n")
    (root / "allocation-map.png").write_bytes(b"\x89PNG\r\n")
    assert artifact_binding(root, launch_model=root)["model_sha"]

    (root / "extra.bin").write_bytes(b"\0")
    with pytest.raises(ValueError, match="differ from finalized inventory"):
        artifact_binding(root, launch_model=root)
    (root / "extra.bin").unlink()

    # This fixture inventories shipcard.json (the real exporter does too —
    # the card occupies a fixed-size reservation, so filling slots never
    # moves its byte count). The byte check therefore still applies to it:
    ship = root / "shipcard.json"
    original_card = ship.read_bytes()
    ship.write_bytes(original_card + b" ")
    with pytest.raises(ValueError, match="differ from finalized inventory"):
        artifact_binding(root, launch_model=root)
    ship.write_bytes(original_card)
    assert artifact_binding(root, launch_model=root)["model_sha"]

    # A NON-inventoried shipcard.json (an artifact class whose card arrives
    # wholly after export) is documentation and is tolerated by name.
    (tmp_path / "post-export-card").mkdir()
    root3, _card3 = _artifact(tmp_path / "post-export-card")
    (root3 / "shipcard.json").unlink()
    quant3 = __import__("json").loads(
        (root3 / "quant_config.json").read_text())
    _finalize_inventory(root3, quant3)
    (root3 / "shipcard.json").write_text("{}\n")
    assert artifact_binding(root3, launch_model=root3)["model_sha"]

    # An INVENTORIED file keeps its byte check even under a documentation
    # name: rebuild the artifact with README.md inside the finalized ledger,
    # then let it drift.
    (tmp_path / "inventoried-readme").mkdir()
    root2, _card2 = _artifact(tmp_path / "inventoried-readme")
    (root2 / "README.md").write_text("shipped at export\n")
    quant = __import__("json").loads(
        (root2 / "quant_config.json").read_text())
    _finalize_inventory(root2, quant)
    assert artifact_binding(root2, launch_model=root2)["model_sha"]
    (root2 / "README.md").write_text("drifted after export\n" * 4)
    with pytest.raises(ValueError, match="differ from finalized inventory"):
        artifact_binding(root2, launch_model=root2)


def test_gold_receipt_accepts_the_publishers_frozen_fd_links_only(tmp_path):
    """The publisher replays this contract on its frozen view, where large
    files are `/proc/self/fd/N` links to its own held descriptors; the freeze
    opens sources O_NOFOLLOW so a real symlink can never reach that view.
    The walk follows exactly that form and refuses every other symlink."""
    import os

    root, _card = _artifact(tmp_path)
    record, _metrics = _gold_record(root)
    kwargs = dict(model_dir=root, require_current_artifact_path=True)
    assert _verify_dsv4_gridbook_gold_contract(
        "gold.kl", record, record["metrics"], **kwargs
    ) == []

    weights = root / "model.safetensors"
    backing = tmp_path / "held-backing.bin"
    backing.write_bytes(weights.read_bytes())
    fd = os.open(backing, os.O_RDONLY)
    try:
        weights.unlink()
        weights.symlink_to(f"/proc/self/fd/{fd}")
        assert _verify_dsv4_gridbook_gold_contract(
            "gold.kl", record, record["metrics"], **kwargs
        ) == []

        weights.unlink()
        weights.symlink_to(backing)  # an ordinary symlink still refuses
        problems = _verify_dsv4_gridbook_gold_contract(
            "gold.kl", record, record["metrics"], **kwargs
        )
        assert any("contains a symlink" in p for p in problems)
    finally:
        os.close(fd)


def test_gold_receipt_survives_move_and_weight_reattest(tmp_path, monkeypatch):
    root, card = _artifact(tmp_path)
    record, _metrics = _gold_record(root)
    assert _verify_dsv4_gridbook_gold_contract(
        "gold.kl",
        record,
        record["metrics"],
        model_dir=root,
        require_current_artifact_path=True,
    ) == []
    card["slots"]["gold.kl"] = record
    write_shipcard(root / "shipcard.json", card)

    relocated = tmp_path / "downloaded-artifact"
    shutil.move(str(root), relocated)
    assert not root.exists()
    reattest_weight_stats(relocated / "shipcard.json", relocated)
    moved_card = load_shipcard(relocated / "shipcard.json")

    assert verify(
        moved_card, model_dir=relocated, required=("gold.kl",)
    ) == []
    strict_fill_time_replay = _verify_dsv4_gridbook_gold_contract(
        "gold.kl",
        moved_card["slots"]["gold.kl"],
        moved_card["slots"]["gold.kl"]["metrics"],
        model_dir=relocated,
        require_current_artifact_path=True,
    )
    assert any("artifact-binding replay" in problem for problem in strict_fill_time_replay)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (
            lambda record, _root: record["metrics"]["serve_manifest"].update(
                gpu_count=2
            ),
            "one-Spark Gridbook runtime",
        ),
        (
            lambda record, _root: record["metrics"]["serve_manifest"][
                "server_process_environment"
            ]["processes"][1]["values"].update(VLLM_USE_DEEP_GEMM="1"),
            "process environment",
        ),
        (
            lambda record, root: record["metrics"]["serve_manifest"][
                "artifact_binding"
            ].update(resolved_path=str(root.parent)),
            "artifact-binding replay",
        ),
        (
            lambda record, _root: record["metrics"]["teacher_evidence"][
                "source_model"
            ].update(resolved_commit="foreign-revision"),
            "teacher source identity",
        ),
        (
            lambda record, _root: record["metrics"].pop("teacher_evidence"),
            "teacher evidence",
        ),
        (
            lambda record, _root: record["metrics"]["teacher_evidence"].update(
                topk_coverage_min=0.50
            ),
            "top-K/tail coverage",
        ),
        (
            lambda record, _root: record["metrics"]["serve_manifest"][
                "producer_identity"
            ].update(git_dirty=True),
            "clean exact commit",
        ),
        (
            lambda record, _root: record["metrics"]["serve_manifest"][
                "gridbook_distribution"
            ]["direct_url"]["vcs_info"].update(commit_id="0" * 40),
            "PEP 610 identity",
        ),
        (
            lambda record, _root: record["metrics"]["serve_manifest"][
                "gridbook_distribution"
            ]["import_origin"].update(
                module_file="/tmp/stale/gridbook/__init__.py"
            ),
            "imported Gridbook origin",
        ),
        (
            lambda record, _root: record["metrics"]["serve_manifest"].update(
                resident_extensions=["prismaquant_cb_v2_ext.so"]
            ),
            "finalized routes",
        ),
    ],
)
def test_gold_replay_rejects_resigned_nested_mutations(
    tmp_path, monkeypatch, mutate, fragment,
):
    root, _card = _artifact(tmp_path)
    record, _metrics = _gold_record(root)
    mutate(record, root)
    _resign(record)

    problems = _verify_dsv4_gridbook_gold_contract(
        "gold.kl", record, record["metrics"], model_dir=root
    )
    assert any(fragment in problem for problem in problems), problems


def test_cli_persists_full_gold_contract_and_refuses_missing_teacher(
    tmp_path, monkeypatch,
):
    root, _card = _artifact(tmp_path)
    record, metrics = _gold_record(root)
    payload = {
        **metrics,
        "model": str(root),
        "spec_decode_detected": False,
        "serve_fingerprint": record["serve_fingerprint"],
        "git_commit": record["git_commit"],
    }
    result_path = tmp_path / "gold.json"
    _write_json(result_path, payload)

    assert shipcard_cli([
        "fill", str(root / "shipcard.json"),
        "--slot", "gold.kl",
        "--record", str(result_path),
        "--model-dir", str(root),
        "--passed",
    ]) == 0
    persisted = load_shipcard(root / "shipcard.json")["slots"]["gold.kl"]
    assert persisted["metrics"]["teacher_evidence"] == metrics[
        "teacher_evidence"
    ]
    assert persisted["metrics"]["serve_manifest"] == metrics["serve_manifest"]
    assert (root / "shipcard.json").stat().st_size == SHIPCARD_RESERVED_BYTES

    payload.pop("teacher_evidence")
    bad_path = tmp_path / "bad-gold.json"
    _write_json(bad_path, payload)
    assert shipcard_cli([
        "fill", str(root / "shipcard.json"),
        "--slot", "gold.kl",
        "--record", str(bad_path),
        "--model-dir", str(root),
        "--passed",
    ]) == 2
