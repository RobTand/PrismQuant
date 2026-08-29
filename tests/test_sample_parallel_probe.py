from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import prismaquant.sample_parallel_probe as producer
from prismaquant.perturbed_x_cache import calibration_data_hash
from prismaquant.sample_parallel_probe import (
    ACTIVATION_CACHE_SHARD_SCHEMA,
    ACTIVATION_PRIORITY_SCHEMA,
    CALIBRATION_SCHEMA,
    ROW_INDICES_SCOPE,
    WORKER_SOURCE_CACHE_RECEIPT_SCHEMA,
    SampleParallelProbeError,
    ActivationPriorityPlanCache,
    activation_cache_shard_stamp,
    activation_scope_receipt,
    build_sample_parallel_run_contract,
    build_rtx4090_qname_census,
    global_activation_row_identity,
    load_calibration_partition,
    load_global_importance_receipt,
    load_local_importance_stats,
    merge_activation_priority_reservoir,
    merge_importance_stats,
    plan_sample_partitions,
    prepare_global_calibration,
    prepare_worker_source_cache,
    select_activation_rows_by_global_priority,
    write_local_importance_stats,
    validate_local_producer_snapshot,
)
from prismaquant.sample_parallel_probe_contract import (
    ACTIVATION_PRIORITY_MAX_ROWS,
    activation_priority_key,
    activation_row_priorities,
    activation_row_priority_scalar,
    validate_activation_priority_domain,
)


EXECUTION_BINDING_SHA256 = "e" * 64


def test_mtp_dispatch_keywords_bind_to_real_runner_signature():
    """Keep the real sample/MTP dispatch from failing after GPU precompute."""
    import prismaquant.incremental_probe as incremental_probe

    tree = ast.parse(inspect.getsource(incremental_probe.main))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_mtp_streaming_shard"
    ]
    assert len(calls) == 1
    assert all(keyword.arg is not None for keyword in calls[0].keywords)
    keyword_values = {
        keyword.arg: object() for keyword in calls[0].keywords
    }
    inspect.signature(
        incremental_probe._run_mtp_streaming_shard
    ).bind(object(), **keyword_values)


def test_run_contract_refuses_missing_streamed_identity_before_source_scan(
    monkeypatch,
):
    monkeypatch.delenv(
        "PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE", raising=False,
    )
    monkeypatch.setattr(
        producer,
        "build_rtx4090_qname_census",
        lambda *_args, **_kwargs: pytest.fail(
            "source census must not start without the required cache"
        ),
    )
    with pytest.raises(
        SampleParallelProbeError, match="requires a validated complete"
    ):
        build_sample_parallel_run_contract(
            model="model",
            dataset="dataset",
            calib_seed=42,
            producer_snapshot_sha256="1" * 64,
            producer_snapshot_commit="2" * 40,
            producer_snapshot_tree="3" * 40,
            container_image_digest="sha256:" + "4" * 64,
        )


def test_trusted_json_loader_rejects_nested_duplicate_member(tmp_path):
    artifact = tmp_path / "contract.json"
    artifact.write_text('{"outer":{"identity":"a","identity":"b"}}')
    with pytest.raises(SampleParallelProbeError, match="duplicate JSON member"):
        producer._load_json_mapping(artifact)


def test_incremental_sample_worker_requires_signed_cover():
    command = [
        sys.executable, "-m", "prismaquant.incremental_probe",
        "--model", "model", "--global-calibration-tensor", "calibration.pt",
        "--sample-partition-index", "0",
        "--sample-run-contract", "run-contract.json",
        "--output", "probe.pkl", "--activation-cache-dir", "act",
        "--work-dir", "work",
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert "--sample-cover is required exactly" in completed.stderr


def test_prepare_worker_source_cache_creates_then_exactly_reuses(
    monkeypatch, tmp_path,
):
    import prismaquant.cost_streaming as cost_streaming
    import prismaquant.gpu_guard as gpu_guard
    import prismaquant.model_profiles as model_profiles

    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "state" / "source-cache.json"
    offload = tmp_path / "offload"
    identity = {"content_sha256": "a" * 64}
    events: list[object] = []

    class _Runner:
        def shutdown(self):
            events.append("shutdown")

    def _build_runner(source, **kwargs):
        events.append(("runner", source, kwargs))
        return _Runner()

    def _build_identity(runner, source, *, identity_cache_path):
        assert isinstance(runner, _Runner)
        events.append(("identity", source))
        Path(identity_cache_path).write_bytes(b"host-local-cache\n")
        return identity

    def _validate(source, path, *, require_complete_checkpoint):
        assert Path(source) == model.resolve()
        assert Path(path).read_bytes() == b"host-local-cache\n"
        assert require_complete_checkpoint is True
        events.append(("validate", Path(path).name))
        return identity

    monkeypatch.setattr(
        gpu_guard, "require_cuda_hot_path",
        lambda *args: torch.device("cpu"),
    )
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: "profile")
    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", _build_runner)
    monkeypatch.setattr(cost_streaming, "build_streamed_model_identity", _build_identity)
    monkeypatch.setattr(
        cost_streaming, "validate_cached_streamed_model_identity", _validate,
    )
    monkeypatch.setattr(
        cost_streaming, "compact_streamed_model_identity",
        lambda value, **_kwargs: {"content_sha256": value["content_sha256"]},
    )

    created = prepare_worker_source_cache(
        model=model.resolve(), output=output.resolve(),
        offload_folder=offload.resolve(),
    )
    assert created["schema"] == WORKER_SOURCE_CACHE_RECEIPT_SCHEMA
    assert created["disposition"] == "created"
    assert created["identity"] == {"content_sha256": "a" * 64}
    assert output.read_bytes() == b"host-local-cache\n"
    assert events.count("shutdown") == 1
    assert events[0][0] == "runner"
    assert events[0][2]["max_cache_slots"] == 1
    assert events[0][2]["prefetch_lookahead"] == 0

    before = list(events)
    reused = prepare_worker_source_cache(
        model=model.resolve(), output=output.resolve(),
        offload_folder=offload.resolve(),
    )
    assert reused["disposition"] == "validated_reuse"
    assert events == [*before, ("validate", output.name)]


def test_prepare_worker_source_cache_refuses_invalid_existing_file(
    monkeypatch, tmp_path,
):
    import prismaquant.cost_streaming as cost_streaming

    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "source-cache.json"
    output.write_text("stale")
    monkeypatch.setattr(
        cost_streaming, "validate_cached_streamed_model_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    monkeypatch.setattr(
        cost_streaming, "build_streamed_causal_lm",
        lambda *_args, **_kwargs: pytest.fail("invalid cache must not be rebuilt"),
    )
    with pytest.raises(SampleParallelProbeError, match="refusing overwrite"):
        prepare_worker_source_cache(
            model=model.resolve(), output=output.resolve(),
            offload_folder=(tmp_path / "offload").resolve(),
        )
    assert output.read_text() == "stale"


def test_partition_plan_is_deterministic_exact_cover():
    parts = plan_sample_partitions(7, 3)
    assert [part.sample_indices for part in parts] == [
        (0, 1, 2), (3, 4), (5, 6),
    ]
    assert [part.sample_start for part in parts] == [0, 3, 5]
    assert [part.sample_stop for part in parts] == [3, 5, 7]
    assert [i for part in parts for i in part.sample_indices] == list(range(7))
    assert plan_sample_partitions(7, 3) == parts


def test_qname_manifests_are_derived_from_strict_source_census(
    monkeypatch, tmp_path,
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    tensors = {
        "model.layers.0.q_proj.weight": {"dtype": "BF16", "shape": [2, 3]},
        "lm_head.weight": {"dtype": "BF16", "shape": [2, 3]},
        "mtp.fc.weight": {"dtype": "BF16", "shape": [2, 6]},
        "model.visual.blocks.0.attn.qkv.weight": {
            "dtype": "BF16", "shape": [2, 3],
        },
    }
    linears = {
        name.removesuffix(".weight"): name for name in tensors
    }
    weight_map = {name: "model.safetensors" for name in tensors}
    import prismaquant.rtx4090_artifact_census as census_module
    import prismaquant.rtx4090_qwen38_policy as policy_module

    monkeypatch.setattr(
        policy_module, "validate_qwen38_dense_config",
        lambda *_a, **_k: {"source_layout": "flattened_text"},
    )
    monkeypatch.setattr(
        census_module, "expected_qwen38_source_layout",
        lambda *_a, **_k: (tensors, linears),
    )
    monkeypatch.setattr(
        census_module, "scan_indexed_safetensors",
        lambda *_a, **_k: (tensors, weight_map),
    )
    source_config_sha = producer._canonical_sha256(
        {}, where="test source config"
    )
    weight_map_sha = producer._canonical_sha256(
        weight_map, where="test weight map"
    )
    shard_manifest = [{
        "name": "model.safetensors", "size": 1, "sha256": "9" * 64,
    }]
    model_body = {
        "schema": producer.SOURCE_MODEL_CONTENT_SCHEMA,
        "derivation": "test_v1", "upstream_content_sha256": None,
        "upstream_portable_content_sha256": None,
        "content_sha256": producer._canonical_sha256({
            "source_config_sha256": source_config_sha,
            "checkpoint_weight_map_sha256": weight_map_sha,
            "shards": shard_manifest,
        }, where="test model content"),
        "resolved_commit": None,
        "checkpoint_tensors": 4, "checkpoint_shards": 1,
        "checkpoint_weight_map_sha256": weight_map_sha,
        "shards": shard_manifest,
    }
    monkeypatch.setattr(
        producer, "_source_model_content_identity",
        lambda *_a, **_k: {
            **model_body,
            "identity_sha256": producer._canonical_sha256(
                model_body, where="test model content"
            ),
        },
    )
    census = build_rtx4090_qname_census(model)
    assert set(census["probe_qname_manifest"]["entries"]) == {
        "model.layers.0.q_proj", "lm_head", "mtp.fc",
    }
    assert set(census["activation_qname_manifest"]["entries"]) == {
        "model.layers.0.q_proj",
    }
    assert set(census["terminal_qname_manifest"]["entries"]) == {
        "lm_head", "mtp.fc",
    }


def test_qname_census_accepts_exact_staged_text_wrapper_namespace(
    monkeypatch, tmp_path,
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    expected = {
        "model.layers.0.q_proj.weight": {
            "dtype": "BF16", "shape": [2, 3],
        },
        "lm_head.weight": {"dtype": "BF16", "shape": [2, 3]},
        "mtp.fc.weight": {"dtype": "BF16", "shape": [2, 6]},
    }
    observed = {
        "model.language_model.layers.0.q_proj.weight": expected[
            "model.layers.0.q_proj.weight"
        ],
        "lm_head.weight": expected["lm_head.weight"],
        "mtp.fc.weight": expected["mtp.fc.weight"],
    }
    linears = {
        name.removesuffix(".weight"): name for name in expected
    }
    raw_weight_map = {name: "model.safetensors" for name in observed}
    import prismaquant.rtx4090_artifact_census as census_module
    import prismaquant.rtx4090_qwen38_policy as policy_module

    monkeypatch.setattr(
        policy_module, "validate_qwen38_dense_config",
        lambda *_a, **_k: {"source_layout": "flattened_text"},
    )
    monkeypatch.setattr(
        census_module, "expected_qwen38_source_layout",
        lambda *_a, **_k: (expected, linears),
    )
    monkeypatch.setattr(
        census_module, "scan_indexed_safetensors",
        lambda *_a, **_k: (observed, raw_weight_map),
    )
    source_config_sha = producer._canonical_sha256(
        {}, where="test source config"
    )
    weight_map_sha = producer._canonical_sha256(
        raw_weight_map, where="test raw source weight map"
    )
    shard_manifest = [{
        "name": "model.safetensors", "size": 1, "sha256": "9" * 64,
    }]
    model_body = {
        "schema": producer.SOURCE_MODEL_CONTENT_SCHEMA,
        "derivation": "test_v1", "upstream_content_sha256": None,
        "upstream_portable_content_sha256": None,
        "content_sha256": producer._canonical_sha256({
            "source_config_sha256": source_config_sha,
            "checkpoint_weight_map_sha256": weight_map_sha,
            "shards": shard_manifest,
        }, where="test model content"),
        "resolved_commit": None,
        "checkpoint_tensors": 3, "checkpoint_shards": 1,
        "checkpoint_weight_map_sha256": weight_map_sha,
        "shards": shard_manifest,
    }
    monkeypatch.setattr(
        producer, "_source_model_content_identity",
        lambda *_a, **_k: {
            **model_body,
            "identity_sha256": producer._canonical_sha256(
                model_body, where="test model content"
            ),
        },
    )

    census = build_rtx4090_qname_census(model)

    source = census["source_census"]
    assert source["source_layout"] == "flattened_text"
    assert source["source_weight_map_sha256"] == weight_map_sha
    assert source["linear_entries"]["model.layers.0.q_proj"][
        "source_tensor"
    ] == "model.layers.0.q_proj.weight"


def test_staged_text_tensor_view_refuses_mixed_body_namespaces():
    with pytest.raises(SampleParallelProbeError, match="mixes model"):
        producer._canonical_staged_text_tensor_view(
            {
                "model.layers.0.q_proj.weight": {
                    "dtype": "BF16", "shape": [2, 3],
                },
                "model.language_model.layers.1.q_proj.weight": {
                    "dtype": "BF16", "shape": [2, 3],
                },
            },
            source_layout="flattened_text",
        )


def test_prepare_artifact_binds_every_local_slice(monkeypatch, tmp_path):
    ids = torch.arange(24, dtype=torch.long).reshape(6, 4)

    class _Tokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", _Tokenizer)
    monkeypatch.setattr(producer, "load_calibration", lambda *_a, **_k: ids)
    artifact = tmp_path / "calibration.pt"
    manifest_path = tmp_path / "calibration.json"
    manifest = prepare_global_calibration(
        model="toy", dataset="cal.jsonl", nsamples=6, seqlen=4,
        calib_seed=42, partition_count=2, output=artifact,
        manifest_output=manifest_path,
    )
    assert manifest["schema"] == CALIBRATION_SCHEMA
    assert json.loads(manifest_path.read_text()) == manifest
    assert len(manifest["partition_contracts"]) == 2

    left, left_contract = load_calibration_partition(
        artifact, partition_index=0,
    )
    right, right_contract = load_calibration_partition(
        artifact, partition_index=1,
    )
    torch.testing.assert_close(torch.cat([left, right]), ids)
    assert left_contract == manifest["partition_contracts"][0]
    assert right_contract == manifest["partition_contracts"][1]
    assert left_contract["global_calibration_hash"] == calibration_data_hash(ids)
    # Crash recovery: an already-published deterministic tensor may be reused
    # to reconstruct/validate its manifest without retokenizing.
    manifest_path.unlink()
    monkeypatch.setattr(
        producer, "load_calibration",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("retokenized")),
    )
    assert prepare_global_calibration(
        model="toy", dataset="cal.jsonl", nsamples=6, seqlen=4,
        calib_seed=42, partition_count=2, output=artifact,
        manifest_output=manifest_path,
    ) == manifest


def _contract(index: int) -> dict[str, object]:
    start = index * 2
    local = torch.arange(start * 4, (start + 2) * 4).reshape(2, 4)
    return {
        "schema": "prismaquant.sample_parallel_probe.partition.v1",
        "global_calibration_hash": "a" * 32,
        "calibration_artifact_sha256": "b" * 64,
        "global_samples": 4,
        "seqlen": 4,
        "partition_count": 2,
        "partition_index": index,
        "sample_indices": list(range(start, start + 2)),
        "sample_start": start,
        "sample_stop": start + 2,
        "local_calibration_hash": calibration_data_hash(local),
        "local_samples": 2,
        "dataset": "cal.jsonl",
        "model": "toy",
        "calib_seed": 42,
    }


def test_two_stage_importance_receipt_exact_cover_and_resume(tmp_path):
    local_paths = [tmp_path / "ce0.json", tmp_path / "ce1.json"]
    for index, path in enumerate(local_paths):
        contract = _contract(index)
        write_local_importance_stats(
            path, partition_contract=contract,
            execution_identity_sha256=EXECUTION_BINDING_SHA256,
            ce_sum=12.0 + index * 6.0, ce_count=6,
        )
        loaded = load_local_importance_stats(
            path, partition_contract=contract,
            execution_identity_sha256=EXECUTION_BINDING_SHA256,
        )
        assert loaded["phase1_reused_across_barrier"] is False

    output = tmp_path / "global-ce.json"
    receipt = merge_importance_stats(list(reversed(local_paths)), output)
    assert receipt["body_global_ce_mean"] == 2.5
    assert receipt["body_global_ce_count"] == 12
    assert receipt["exact_sample_cover"] is True
    assert receipt["bitwise_monolithic_equivalence_claimed"] is False
    assert receipt["barrier_execution"] == "duplicate_phase1_forward_v1"
    loaded = load_global_importance_receipt(
        output, partition_contract=_contract(0),
        execution_identity_sha256=EXECUTION_BINDING_SHA256,
    )
    assert loaded == receipt


def test_importance_receipt_rejects_inexact_shifted_token_count(tmp_path):
    with pytest.raises(SampleParallelProbeError, match="shifted-token cover"):
        write_local_importance_stats(
            tmp_path / "bad.json", partition_contract=_contract(0),
            execution_identity_sha256=EXECUTION_BINDING_SHA256,
            ce_sum=1.0, ce_count=5,
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda receipt: receipt["local_receipts"][0].update(
                {"phase1_reused_across_barrier": True}
            ),
            "digest differs",
        ),
        (
            lambda receipt: receipt.update(
                {"calibration_artifact_sha256": "c" * 64}
            ),
            "CE/calibration invariant differs",
        ),
    ],
)
def test_global_importance_receipt_tamper_is_fail_closed(tmp_path, mutate, match):
    paths = []
    for index in range(2):
        path = tmp_path / f"local-{index}.json"
        write_local_importance_stats(
            path, partition_contract=_contract(index), ce_sum=6.0,
            execution_identity_sha256=EXECUTION_BINDING_SHA256,
            ce_count=6,
        )
        paths.append(path)
    output = tmp_path / "global.json"
    merge_importance_stats(paths, output)
    receipt = json.loads(output.read_text())
    mutate(receipt)
    body = dict(receipt)
    body.pop("receipt_sha256")
    receipt["receipt_sha256"] = producer._canonical_sha256(
        body, where="resealed malicious global receipt"
    )
    output.write_text(json.dumps(receipt))
    with pytest.raises(SampleParallelProbeError, match=match):
        load_global_importance_receipt(
            output,
            partition_contract=_contract(0),
            execution_identity_sha256=EXECUTION_BINDING_SHA256,
        )


def test_priority_reservoir_matches_one_shot_and_keeps_local_indices():
    contract = _contract(1)
    inputs = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    local_rows = torch.arange(8, dtype=torch.long)
    one_x, one_rows, one_priorities = select_activation_rows_by_global_priority(
        inputs, local_rows, qname="model.layers.0.q_proj",
        partition_contract=contract, rows_limit=3,
    )
    first = merge_activation_priority_reservoir(
        prior_inputs=None, prior_local_rows=None, prior_priorities=None,
        new_inputs=inputs[:4], new_local_rows=local_rows[:4],
        qname="model.layers.0.q_proj", partition_contract=contract,
        rows_limit=3,
    )
    chunked = merge_activation_priority_reservoir(
        prior_inputs=first[0], prior_local_rows=first[1],
        prior_priorities=first[2], new_inputs=inputs[4:],
        new_local_rows=local_rows[4:], qname="model.layers.0.q_proj",
        partition_contract=contract, rows_limit=3,
    )
    torch.testing.assert_close(chunked[0], one_x)
    torch.testing.assert_close(chunked[1], one_rows)
    torch.testing.assert_close(chunked[2], one_priorities)
    assert int(one_rows.max()) < 8
    global_rows = global_activation_row_identity(one_rows, contract)
    torch.testing.assert_close(global_rows, one_rows + 8)


def test_activation_priority_known_vectors_and_uint32_boundaries():
    qname = "model.layers.0.q_proj"
    calibration_hash = "a" * 32
    rows = [
        0, 1, 65535, 65536, (1 << 31) - 1, 1 << 31,
        ACTIVATION_PRIORITY_MAX_ROWS - 1,
    ]
    assert activation_priority_key(calibration_hash, qname) == (
        2261972762, 2393839975,
    )
    expected = [
        484211665, 827814195, 36613847, 1464622029,
        3455518833, 2398396980, 2952284691,
    ]
    assert [
        activation_row_priority_scalar(calibration_hash, qname, row)
        for row in rows
    ] == expected
    observed = activation_row_priorities(
        calibration_hash, qname, torch.tensor(rows, dtype=torch.long)
    )
    torch.testing.assert_close(observed, torch.tensor(expected, dtype=torch.long))


def test_activation_priority_is_a_permutation_on_dense_prefix():
    rows = torch.arange(1 << 16, dtype=torch.long)
    priorities = activation_row_priorities(
        "a" * 32, "model.layers.3.self_attn.q_proj", rows
    )
    assert int(torch.unique(priorities).numel()) == int(rows.numel())


def test_activation_priority_domain_refuses_zero_and_over_uint32():
    assert validate_activation_priority_domain(1 << 20, 1 << 12) == 1 << 32
    with pytest.raises(ValueError, match="0 < global_samples"):
        validate_activation_priority_domain(0, 4)
    with pytest.raises(ValueError, match=r"2\*\*32"):
        validate_activation_priority_domain((1 << 32) + 1, 1)
    with pytest.raises(ValueError, match="uint32"):
        activation_row_priority_scalar(
            "a" * 32, "model.layers.0.q_proj", 1 << 32
        )


def test_fused_siblings_reuse_one_cached_device_selection_plan():
    cache = ActivationPriorityPlanCache(_contract(0))
    q_rows, q_priorities = cache.top_rows(
        "model.layers.0.self_attn.q_proj",
        device=torch.device("cpu"), rows_limit=3,
    )
    k_rows, k_priorities = cache.top_rows(
        "model.layers.0.self_attn.k_proj",
        device=torch.device("cpu"), rows_limit=3,
    )
    assert q_rows.data_ptr() == k_rows.data_ptr()
    assert q_priorities.data_ptr() == k_priorities.data_ptr()
    cache.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host required")
def test_activation_priority_cpu_cuda_and_no_sync_parity():
    rows = torch.tensor(
        [0, 1, 65535, 65536, (1 << 31), (1 << 32) - 1],
        dtype=torch.long,
    )
    expected = activation_row_priorities(
        "a" * 32, "model.layers.0.self_attn.q_proj", rows
    )
    rows_cuda = rows.cuda()
    old_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        observed = activation_row_priorities(
            "a" * 32,
            "model.layers.0.self_attn.q_proj",
            rows_cuda,
        )
        selected = torch.topk(
            observed, 3, largest=False, sorted=True
        ).indices
    finally:
        torch.cuda.set_sync_debug_mode(old_mode)
    torch.testing.assert_close(observed.cpu(), expected)
    assert selected.device.type == "cuda"


def test_local_ce_resume_is_bound_to_execution_identity(tmp_path):
    path = tmp_path / "local.json"
    write_local_importance_stats(
        path,
        partition_contract=_contract(0),
        execution_identity_sha256=EXECUTION_BINDING_SHA256,
        ce_sum=6.0,
        ce_count=6,
    )
    with pytest.raises(SampleParallelProbeError, match="execution identity"):
        load_local_importance_stats(
            path,
            partition_contract=_contract(0),
            execution_identity_sha256="f" * 64,
        )


def test_local_producer_snapshot_rehash_refuses_manifested_file_tamper(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    (source / "prismaquant").mkdir(parents=True)
    (source / "tools").mkdir()
    (source / "prismaquant" / "__init__.py").write_text("VALUE = 1\n")
    (source / "prismaquant" / "sample_parallel_probe.py").write_text(
        "# producer\n"
    )
    (source / "prismaquant" / "incremental_probe.py").write_text(
        "# worker\n"
    )
    (source / "tools" / "container_runtime_identity.py").write_text("# tool\n")
    (source / "tools" / "prismaquant_source_bootstrap.py").write_text(
        "# bootstrap\n"
    )
    (source / "tools" / "prismaquant_runtime_snapshot.py").write_text(
        (repository / "tools" / "prismaquant_runtime_snapshot.py").read_text()
    )
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(source), "-c", "user.name=test",
        "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture",
    ], check=True)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    tool = repository / "tools" / "prismaquant_runtime_snapshot.py"
    created = subprocess.run([
        sys.executable, str(tool), "materialize",
        "--source-root", str(source),
        "--cache-root", str(tmp_path / "cache"),
        "--commit", commit,
    ], check=True, capture_output=True, text=True)
    snapshot = json.loads(created.stdout)
    verified = validate_local_producer_snapshot(
        snapshot["snapshot"],
        expected_closure_sha256=snapshot["closure_sha256"],
        expected_commit=snapshot["commit"],
        expected_tree=snapshot["tree"],
        require_current_module_inside=False,
    )
    assert verified["closure_sha256"] == snapshot["closure_sha256"]
    (Path(snapshot["snapshot"]) / "prismaquant" / "__init__.py").write_text(
        "VALUE = 2\n"
    )
    # Replacing the candidate verifier with an always-success function cannot
    # hide the other tamper: validation never executes snapshot-owned code.
    (Path(snapshot["snapshot"]) / "tools" / "prismaquant_runtime_snapshot.py").write_text(
        "def verify_snapshot(*args, **kwargs):\n"
        "    return {'closure_sha256': kwargs['expected_closure_sha256']}\n"
    )
    with pytest.raises(SampleParallelProbeError, match="files differ"):
        validate_local_producer_snapshot(
            snapshot["snapshot"],
            expected_closure_sha256=snapshot["closure_sha256"],
            expected_commit=snapshot["commit"],
            expected_tree=snapshot["tree"],
            require_current_module_inside=False,
        )


def test_stage1_ce_postflight_failure_publishes_no_receipt(tmp_path):
    from prismaquant.incremental_probe import (
        _publish_sample_parallel_importance_stats,
    )

    output = tmp_path / "ce.json"

    def _fail_postflight():
        raise RuntimeError("source changed after phase 1")

    with pytest.raises(RuntimeError, match="source changed"):
        _publish_sample_parallel_importance_stats(
            output,
            partition_contract=_contract(0),
            execution_identity_sha256=EXECUTION_BINDING_SHA256,
            ce_sum=6.0,
            ce_count=6,
            publication_postflight=_fail_postflight,
        )
    assert not output.exists()


@pytest.mark.parametrize("bad_field", ["ids", "router"])
def test_sample_precompute_reuse_refuses_wrong_ids_and_routed_maps(
    tmp_path, bad_field,
):
    from prismaquant.incremental_probe import _load_precompute_cache

    contract = _contract(0)
    expected_ids = torch.arange(8, dtype=torch.long).reshape(2, 4)
    payload = {
        "meta": {"sample_parallel": contract},
        "ids_cpu": expected_ids.clone(),
        "expert_info": {},
        "router_counts": {},
        "router_totals": {},
        "router_active_counts": {},
        "expert_route_stats": {},
    }
    if bad_field == "ids":
        payload["ids_cpu"] = expected_ids.flip(1)
    else:
        payload["router_counts"] = {"router": {"0": 1.0}}
    path = tmp_path / f"precompute-{bad_field}.pt"
    torch.save(payload, path)
    assert _load_precompute_cache(
        path,
        {"sample_parallel": contract},
        torch.device("cpu"),
        sample_calibration=expected_ids,
    ) is None


def test_sample_mode_disables_same_path_process_global_model_reuse(monkeypatch):
    from prismaquant.incremental_probe import _persistent_probe_context_enabled

    monkeypatch.setenv("PRISMAQUANT_PROBE_CTX_CACHE", "1")
    assert _persistent_probe_context_enabled(None) is True
    assert _persistent_probe_context_enabled(_contract(0)) is False


def test_divergent_duplicate_linear_cache_rows_refuse_synthesized_reuse(tmp_path):
    import pickle
    from prismaquant.incremental_probe import scan_cached_linear_stats

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for index, raw in enumerate((1.0, 2.0)):
        with (shard_dir / f"probe_shard_{index:03d}.pkl").open("wb") as handle:
            pickle.dump({
                "stats": {
                    "model.layers.0.q_proj": {
                        "h_trace_raw": raw,
                        "fisher_row": np.array([raw], dtype=np.float32),
                    },
                },
                "meta": {},
            }, handle)
    assert scan_cached_linear_stats(shard_dir, {}) == {}


@pytest.mark.parametrize("mode", ["empty", "unmatched", "unset"])
def test_sample_mtp_source_coverage_fails_before_any_forward(mode):
    from prismaquant.incremental_probe import _load_mtp_source_for_probe

    class _Profile:
        name = "fixture"
        load_calls = 0

        @staticmethod
        def mtp_source_prefix():
            return "mtp."

        @staticmethod
        def read_mtp_source_state_dict(_path):
            return {} if mode == "empty" else {"weight": torch.ones(1)}

        def load_mtp_state_dict(self, _module, _raw):
            self.load_calls += 1
            if mode == "unmatched":
                return ["weight"], []
            if mode == "unset":
                return [], ["projection.weight"]
            return [], []

    profile = _Profile()
    with pytest.raises(RuntimeError, match="no source|coverage failure"):
        _load_mtp_source_for_probe(
            profile, torch.nn.Linear(1, 1), "model",
            sample_parallel=True,
        )
    assert profile.load_calls == (0 if mode == "empty" else 1)


def test_lm_head_linear_hook_observes_exact_t_minus_one_reference_rows():
    from prismaquant.incremental_probe import _scored_lm_head_logits

    torch.manual_seed(0)
    batch, seqlen, hidden_size, vocab = 3, 7, 5, 11
    hidden = torch.randn(
        batch, seqlen, hidden_size, requires_grad=True
    )
    head = torch.nn.Linear(hidden_size, vocab, bias=False)
    observed_rows: list[int] = []
    saved_inputs: list[torch.Tensor] = []
    stats = {
        "n_tokens_seen": 0,
        "act_sq_sum": torch.zeros(hidden_size, dtype=torch.float32),
        "act_absmax": torch.zeros(hidden_size, dtype=torch.float32),
    }

    def _forward_hook(_module, inputs, _output):
        value = inputs[0].detach()
        observed_rows.append(int(value.numel() // hidden_size))
        saved_inputs.append(value)

    def _backward_hook(_module, _grad_input, _grad_output):
        value = saved_inputs.pop().reshape(-1, hidden_size)
        stats["n_tokens_seen"] += int(value.shape[0])
        stats["act_sq_sum"].add_(
            value.pow(2).sum(dim=0, dtype=torch.float32)
        )
        torch.maximum(
            stats["act_absmax"],
            torch.maximum(
                value.amax(dim=0).abs(), value.amin(dim=0).abs()
            ).to(torch.float32),
            out=stats["act_absmax"],
        )

    handles = [
        head.register_forward_hook(_forward_hook),
        head.register_full_backward_hook(_backward_hook),
    ]
    try:
        actual = _scored_lm_head_logits(
            head, hidden, start=0, scored_tokens=seqlen - 1
        )
        actual.sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    expected_hidden = hidden.detach()[:, :-1, :]
    expected = head(expected_hidden).float()
    expected_flat = expected_hidden.reshape(-1, hidden_size)
    torch.testing.assert_close(actual, expected)
    assert observed_rows == [batch * (seqlen - 1)]
    assert stats["n_tokens_seen"] == batch * (seqlen - 1)
    torch.testing.assert_close(
        stats["act_sq_sum"],
        expected_flat.pow(2).sum(dim=0, dtype=torch.float32),
    )
    torch.testing.assert_close(
        stats["act_absmax"],
        torch.maximum(
            expected_flat.amax(dim=0).abs(),
            expected_flat.amin(dim=0).abs(),
        ).to(torch.float32),
    )


def test_activation_shard_stamp_is_strict_dense_local_scope():
    stamp = activation_cache_shard_stamp(
        _contract(0), qname="model.layers.0.self_attn.q_proj",
        rows_limit=256, candidate_rows=8,
        execution_identity_sha256=EXECUTION_BINDING_SHA256,
    )
    assert stamp == {
        "schema": ACTIVATION_CACHE_SHARD_SCHEMA,
        "priority_schema": ACTIVATION_PRIORITY_SCHEMA,
        "priority_group": "model.layers.0.qkv",
        "selection": "local_top_r",
        "row_indices_scope": ROW_INDICES_SCOPE,
        "layout": "dense_linear",
        "global_calibration_hash": "a" * 32,
        "execution_identity_sha256": EXECUTION_BINDING_SHA256,
        "partition_index": 0,
        "rows_limit": 256,
        "candidate_rows": 8,
    }
    scope = activation_scope_receipt()
    assert scope["resident_lm_head_cache"] == "omitted_terminal_bf16"
    assert scope["mtp_cache"] == "omitted_no_direct_sample_token_row_map_v1"
    assert scope["separate_probe_and_activation_qname_manifests_required"] is True
