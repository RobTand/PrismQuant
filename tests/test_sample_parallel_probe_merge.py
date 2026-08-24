import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

import prismaquant.sample_parallel_probe_merge as merge_module
import prismaquant.sample_parallel_probe as producer
from prismaquant.sample_parallel_probe import (
    BODY_STATS_AND_ACTIVATION,
    BODY_TOKEN_ROWS,
    LM_HEAD_STATS_ONLY,
    LM_HEAD_TOKEN_ROWS,
    MTP_STATS_ONLY,
    MTP_TOKEN_ROWS,
    build_execution_identity,
    validate_qname_census,
    write_local_importance_stats,
    merge_importance_stats,
)
from prismaquant.sample_parallel_probe_merge import (
    ACTIVATION_CACHE_SHARD_SCHEMA,
    ACTIVATION_PRIORITY_SCHEMA,
    ACTIVATION_CACHE_MANIFEST,
    SAMPLE_PARALLEL_SHARD_SCHEMA,
    SampleParallelMergeError,
    activation_priority_group,
    activation_row_priorities,
    build_sample_parallel_cover,
    merge_sample_parallel_activation_caches,
    merge_sample_parallel_probe_payloads,
    validate_merged_activation_cache_output,
    validate_worker_sample_cover,
)
from prismaquant.rtx4090_qwen38_policy import (
    RTX4090_QWEN38_POLICY_ID,
    RTX4090_QWEN38_POLICY_SCHEMA,
)


BODY_QNAMES = ("model.layers.0.q_proj",)
PROBE_QNAMES = (*BODY_QNAMES, "lm_head", "mtp.fc")
PRODUCER_IDENTITY = {
    "producer_snapshot_sha256": "4" * 64,
    "producer_snapshot_commit": "5" * 40,
    "producer_snapshot_tree": "6" * 40,
    "container_image_digest": "sha256:" + "7" * 64,
}


def _qname_census(
    *,
    shard_sha256: str = "9" * 64,
    derivation: str = "test_v1",
    upstream_content_sha256: str | None = None,
    upstream_portable_content_sha256: str | None = None,
    model: str = "model-content-id",
    body_qnames: tuple[str, ...] = BODY_QNAMES,
):
    entries = {
        name: {
            "source_tensor": name + ".weight",
            "source_dtype": "BF16", "shape": [2, 3],
            "disposition": BODY_STATS_AND_ACTIVATION,
            "token_rows_per_sample": BODY_TOKEN_ROWS,
            "terminal_format": None,
        }
        for name in body_qnames
    }
    entries.update({
        "lm_head": {
            "source_tensor": "lm_head.weight",
            "source_dtype": "BF16", "shape": [2, 3],
            "disposition": LM_HEAD_STATS_ONLY,
            "token_rows_per_sample": LM_HEAD_TOKEN_ROWS,
            "terminal_format": "BF16",
        },
        "mtp.fc": {
            "source_tensor": "mtp.fc.weight",
            "source_dtype": "BF16", "shape": [2, 3],
            "disposition": MTP_STATS_ONLY,
            "token_rows_per_sample": MTP_TOKEN_ROWS,
            "terminal_format": "BF16",
        },
    })
    shard = {
        "name": "model.safetensors", "size": 1,
        "sha256": shard_sha256,
    }
    content_sha = producer._canonical_sha256({
        "source_config_sha256": "1" * 64,
        "checkpoint_weight_map_sha256": "3" * 64,
        "shards": [shard],
    }, where="test model content")
    model_body = {
        "schema": producer.SOURCE_MODEL_CONTENT_SCHEMA,
        "derivation": derivation,
        "upstream_content_sha256": upstream_content_sha256,
        "upstream_portable_content_sha256": (
            upstream_portable_content_sha256
        ),
        "content_sha256": content_sha, "resolved_commit": None,
        "checkpoint_tensors": len(entries), "checkpoint_shards": 1,
        "checkpoint_weight_map_sha256": "3" * 64,
        "shards": [shard],
    }
    model_identity = {
        **model_body,
        "identity_sha256": producer._canonical_sha256(
            model_body, where="test model identity"
        ),
    }
    source_body = {
        "schema": producer.SOURCE_QNAME_CENSUS_SCHEMA,
        "model": model,
        "producer_profile_schema": RTX4090_QWEN38_POLICY_SCHEMA,
        "producer_profile_id": RTX4090_QWEN38_POLICY_ID,
        "source_layout": "flattened_text",
        "source_config_sha256": "1" * 64,
        "source_tensor_manifest_sha256": "2" * 64,
        "source_weight_map_sha256": "3" * 64,
        "source_tensor_count": len(entries),
        "source_linear_count": len(entries),
        "source_model_identity": model_identity,
        "linear_entries": entries,
    }
    source = {
        **source_body,
        "identity_sha256": producer._canonical_sha256(
            source_body, where="test source census"
        ),
    }
    source_sha = source["identity_sha256"]
    probe = producer._manifest_with_digest(
        kind="full_probe", source_census_sha256=source_sha, entries=entries,
    )
    activation = producer._manifest_with_digest(
        kind="dense_body_activation", source_census_sha256=source_sha,
        entries={name: entries[name] for name in body_qnames},
    )
    terminal = producer._manifest_with_digest(
        kind="terminal_bf16_stats_only", source_census_sha256=source_sha,
        entries={name: entries[name] for name in ("lm_head", "mtp.fc")},
    )
    body = {
        "source_census": source,
        "probe_qname_manifest": probe,
        "activation_qname_manifest": activation,
        "terminal_qname_manifest": terminal,
    }
    return validate_qname_census({
        **body,
        "identity_sha256": producer._canonical_sha256(
            body, where="test qname census"
        ),
    })


def _cover(*, body_qnames: tuple[str, ...] = BODY_QNAMES):
    census = _qname_census(body_qnames=body_qnames)
    execution = build_execution_identity(
        model="model-content-id", dataset="test", calib_seed=42,
        dtype="bf16", importance_weighting=False, emit_marginals=True,
        activation_rows_limit=1024, qname_census=census,
        **PRODUCER_IDENTITY,
    )
    return build_sample_parallel_cover(
        [
            {
                "schema": SAMPLE_PARALLEL_SHARD_SCHEMA,
                "global_calibration_hash": "a" * 32,
                "calibration_artifact_sha256": "b" * 64,
                "global_samples": 4,
                "seqlen": 4,
                "partition_count": 2,
                "partition_index": 0,
                "sample_indices": [0, 1],
                "sample_start": 0,
                "sample_stop": 2,
                "local_calibration_hash": "0" * 32,
                "local_samples": 2,
                "dataset": "test",
                "model": "model-content-id",
                "calib_seed": 42,
            },
            {
                "schema": SAMPLE_PARALLEL_SHARD_SCHEMA,
                "global_calibration_hash": "a" * 32,
                "calibration_artifact_sha256": "b" * 64,
                "global_samples": 4,
                "seqlen": 4,
                "partition_count": 2,
                "partition_index": 1,
                "sample_indices": [2, 3],
                "sample_start": 2,
                "sample_stop": 4,
                "local_calibration_hash": "1" * 32,
                "local_samples": 2,
                "dataset": "test",
                "model": "model-content-id",
                "calib_seed": 42,
            },
        ],
        execution_identity=execution,
        qname_census=census,
    )


def _run_contract_for_cover(cover):
    body = {
        "schema": producer.RUN_CONTRACT_SCHEMA,
        "execution_identity": copy.deepcopy(cover["execution_identity"]),
        "qname_census": copy.deepcopy(cover["qname_census"]),
    }
    return producer.validate_run_contract({
        **body,
        "identity_sha256": producer._canonical_sha256(
            body, where="test run contract"
        ),
    })


def _sample_precompute_fixture(tmp_path):
    from prismaquant import incremental_probe as ip
    from prismaquant.perturbed_x_cache import calibration_data_hash

    ids = torch.arange(8, dtype=torch.int64).reshape(2, 4)
    contract = copy.deepcopy(_cover()["shards"][0])
    contract["local_calibration_hash"] = calibration_data_hash(ids)
    meta = {
        "sample_parallel": contract,
        "sample_parallel_qname_census": _qname_census(),
    }
    h_full = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32,
    )
    resident_stats = {
        "lm_head": {
            "h_trace_raw": 21.0,
            "h_w2_sum_raw": 7.0,
            "w_max_abs": 2.0,
            "w_norm_sq": 5.0,
            "n_params": 6,
            "in_features": 3,
            "out_features": 2,
            "n_tokens_seen": 6,
            "route_prob": None,
            "router_path": None,
            "expert_id": None,
            "fisher_row": np.array([6.0, 15.0], dtype=np.float32),
            "fisher_col": np.array([5.0, 7.0, 9.0], dtype=np.float32),
            "g_sq_sum": np.array([10.0, 11.0], dtype=np.float32),
            "act_sq_sum": np.array([8.0, 9.0, 10.0], dtype=np.float32),
            "act_absmax": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        },
    }
    precompute = ip.GlobalPrecompute(
        activations_cpu=[
            torch.zeros(2, 4, 3, dtype=torch.bfloat16),
            torch.ones(2, 4, 3, dtype=torch.bfloat16),
        ],
        grad_at_tail=torch.ones(2, 4, 3, dtype=torch.bfloat16),
        ids=ids,
        resident_stats=resident_stats,
        resident_h_full={"lm_head": h_full},
        resident_g2_per_token={
            "lm_head": torch.arange(1, 7, dtype=torch.float32),
        },
        resident_act_snaps={},
        resident_act_row_indices={},
        expert_info={},
        router_counts={},
        router_totals={},
        router_active_counts={},
        expert_route_stats={},
        shared_pass_state=None,
    )
    path = tmp_path / "precomputed.pt"
    ip._save_precompute_cache(path, precompute, meta)
    return path, meta, ids


def _mutate_sample_precompute_payload(payload, mutation):
    if mutation == "minimal_stat":
        payload["resident_stats"]["lm_head"] = {"h_trace_raw": 21.0}
    elif mutation == "missing_map_entry":
        payload["resident_h_full"].clear()
    elif mutation == "extra_map_entry":
        payload["resident_g2_per_token"]["mtp.fc"] = torch.ones(
            6, dtype=torch.float32
        )
    elif mutation == "h_dtype":
        payload["resident_h_full"]["lm_head"] = payload[
            "resident_h_full"
        ]["lm_head"].to(torch.float64)
    elif mutation == "h_shape":
        payload["resident_h_full"]["lm_head"] = torch.ones(
            3, 2, dtype=torch.float32
        )
    elif mutation == "h_nonfinite":
        payload["resident_h_full"]["lm_head"][0, 0] = float("nan")
    elif mutation == "g2_dtype":
        payload["resident_g2_per_token"]["lm_head"] = payload[
            "resident_g2_per_token"
        ]["lm_head"].to(torch.float64)
    elif mutation == "g2_shape":
        payload["resident_g2_per_token"]["lm_head"] = torch.ones(
            5, dtype=torch.float32
        )
    elif mutation == "g2_nonfinite":
        payload["resident_g2_per_token"]["lm_head"][0] = float("inf")
    elif mutation == "marginal_dtype":
        payload["resident_stats"]["lm_head"]["fisher_row"] = np.array(
            [6.0, 15.0], dtype=np.float64
        )
    elif mutation == "marginal_shape":
        payload["resident_stats"]["lm_head"]["fisher_col"] = np.ones(
            2, dtype=np.float32
        )
    elif mutation == "marginal_nonfinite":
        payload["resident_stats"]["lm_head"]["g_sq_sum"][0] = np.nan
    elif mutation == "trace_inconsistent":
        payload["resident_stats"]["lm_head"]["fisher_row"][0] += 1.0
    elif mutation == "token_type":
        payload["resident_stats"]["lm_head"]["n_tokens_seen"] = True
    elif mutation == "resident_activation":
        payload["resident_act_snaps"]["lm_head"] = [torch.ones(1, 3)]
    elif mutation == "shared_state":
        payload["shared_pass_state"] = {}
    else:  # pragma: no cover - keeps the table closed
        raise AssertionError(mutation)


def test_sample_precompute_cache_valid_resident_payload_round_trip(tmp_path):
    from prismaquant import incremental_probe as ip

    path, meta, ids = _sample_precompute_fixture(tmp_path)
    loaded = ip._load_precompute_cache(
        path, meta, torch.device("cpu"), sample_calibration=ids,
    )

    assert loaded is not None
    assert set(loaded.resident_stats) == {"lm_head"}
    assert loaded.resident_stats["lm_head"]["n_tokens_seen"] == 6
    assert loaded.resident_h_full["lm_head"].dtype == torch.float32
    assert loaded.resident_g2_per_token["lm_head"].shape == (6,)


@pytest.mark.parametrize("mutation", [
    "minimal_stat",
    "missing_map_entry",
    "extra_map_entry",
    "h_dtype",
    "h_shape",
    "h_nonfinite",
    "g2_dtype",
    "g2_shape",
    "g2_nonfinite",
    "marginal_dtype",
    "marginal_shape",
    "marginal_nonfinite",
    "trace_inconsistent",
    "token_type",
    "resident_activation",
    "shared_state",
])
def test_sample_precompute_cache_refuses_malformed_resident_payload(
    tmp_path, mutation,
):
    from prismaquant import incremental_probe as ip

    path, meta, ids = _sample_precompute_fixture(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _mutate_sample_precompute_payload(payload, mutation)
    torch.save(payload, path)

    assert ip._load_precompute_cache(
        path, meta, torch.device("cpu"), sample_calibration=ids,
    ) is None


def test_worker_cover_preflight_binds_run_census_and_selected_calibration():
    cover = _cover()
    run_contract = _run_contract_for_cover(cover)
    assert validate_worker_sample_cover(
        cover,
        run_contract=run_contract,
        partition_contract=cover["shards"][1],
    )["identity_sha256"] == cover["identity_sha256"]

    changed_partition = copy.deepcopy(cover["shards"][1])
    changed_partition["calibration_artifact_sha256"] = "c" * 64
    with pytest.raises(SampleParallelMergeError, match="calibration artifact"):
        validate_worker_sample_cover(
            cover,
            run_contract=run_contract,
            partition_contract=changed_partition,
        )

    changed_execution = build_execution_identity(
        model="model-content-id", dataset="test", calib_seed=42,
        dtype="fp16", importance_weighting=False, emit_marginals=True,
        activation_rows_limit=1024, qname_census=cover["qname_census"],
        **PRODUCER_IDENTITY,
    )
    changed_body = {
        "schema": producer.RUN_CONTRACT_SCHEMA,
        "execution_identity": changed_execution,
        "qname_census": cover["qname_census"],
    }
    changed_run = producer.validate_run_contract({
        **changed_body,
        "identity_sha256": producer._canonical_sha256(
            changed_body, where="changed test run contract"
        ),
    })
    with pytest.raises(SampleParallelMergeError, match="execution identity"):
        validate_worker_sample_cover(
            cover,
            run_contract=changed_run,
            partition_contract=cover["shards"][0],
        )


def test_merged_activation_manifest_rejects_duplicate_json_member(tmp_path):
    output = tmp_path / "merged"
    output.mkdir()
    (output / ACTIVATION_CACHE_MANIFEST).write_text(
        '{"schema":"first","schema":"second"}'
    )
    with pytest.raises(SampleParallelMergeError, match="duplicate JSON member"):
        validate_merged_activation_cache_output(
            output, expected_cover=_cover()
        )


def test_stable_source_projection_ignores_only_host_local_provenance():
    cached = _qname_census(
        derivation="validated_streamed_model_identity_cache_v1",
        upstream_content_sha256="8" * 64,
        upstream_portable_content_sha256="a" * 64,
    )
    other_host = _qname_census(
        derivation="validated_streamed_model_identity_cache_v1",
        upstream_content_sha256="b" * 64,
        upstream_portable_content_sha256="a" * 64,
        model="/another/host/model",
    )
    assert producer.stable_source_census_projection(cached) == (
        producer.stable_source_census_projection(other_host)
    )
    changed = _qname_census(shard_sha256="a" * 64)
    assert producer.stable_source_census_projection(changed) != (
        producer.stable_source_census_projection(cached)
    )


def test_cached_source_census_requires_valid_upstream_streamed_identity():
    with pytest.raises(
        producer.SampleParallelProbeError, match="model-content identity"
    ):
        _qname_census(
            derivation="validated_streamed_model_identity_cache_v1",
            upstream_content_sha256=None,
        )
    with pytest.raises(
        producer.SampleParallelProbeError, match="model-content identity"
    ):
        _qname_census(upstream_content_sha256="not-a-sha256")


def test_worker_local_source_preflight_refuses_changed_stable_census(
    monkeypatch,
):
    expected = _qname_census()
    changed = _qname_census(shard_sha256="a" * 64)
    monkeypatch.setenv(
        "PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE", "/host/local/cache.json"
    )
    monkeypatch.setattr(
        producer, "build_rtx4090_qname_census", lambda _model: changed
    )
    with pytest.raises(
        producer.SampleParallelProbeError, match="stable source census differs"
    ):
        producer.validate_worker_local_source_census(
            "model-content-id", expected
        )


def test_worker_local_source_preflight_refuses_changed_portable_upstream(
    monkeypatch,
):
    expected = _qname_census(
        derivation="validated_streamed_model_identity_cache_v1",
        upstream_content_sha256="8" * 64,
        upstream_portable_content_sha256="a" * 64,
    )
    changed = _qname_census(
        derivation="validated_streamed_model_identity_cache_v1",
        upstream_content_sha256="9" * 64,
        upstream_portable_content_sha256="b" * 64,
        model="/another/host/model",
    )
    monkeypatch.setenv(
        "PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE", "/host/local/cache.json"
    )
    monkeypatch.setattr(
        producer, "build_rtx4090_qname_census", lambda _model: changed
    )
    with pytest.raises(
        producer.SampleParallelProbeError, match="stable source census differs"
    ):
        producer.validate_worker_local_source_census(
            "model-content-id", expected
        )


def test_importance_binding_changes_with_dtype_content_and_producer_snapshot():
    census = _qname_census()
    base = build_execution_identity(
        model="model-content-id", dataset="test", calib_seed=42,
        dtype="bf16", importance_weighting=True, emit_marginals=True,
        activation_rows_limit=1024, qname_census=census,
        **PRODUCER_IDENTITY,
    )
    fp16 = build_execution_identity(
        model="model-content-id", dataset="test", calib_seed=42,
        dtype="fp16", importance_weighting=True, emit_marginals=True,
        activation_rows_limit=1024, qname_census=census,
        **PRODUCER_IDENTITY,
    )
    changed_census = _qname_census(shard_sha256="a" * 64)
    changed_content = build_execution_identity(
        model="model-content-id", dataset="test", calib_seed=42,
        dtype="bf16", importance_weighting=True, emit_marginals=True,
        activation_rows_limit=1024, qname_census=changed_census,
        **PRODUCER_IDENTITY,
    )
    changed_producer_values = dict(PRODUCER_IDENTITY)
    changed_producer_values["producer_snapshot_sha256"] = "8" * 64
    changed_producer = build_execution_identity(
        model="model-content-id", dataset="test", calib_seed=42,
        dtype="bf16", importance_weighting=True, emit_marginals=True,
        activation_rows_limit=1024, qname_census=census,
        **changed_producer_values,
    )
    identities = {
        producer.importance_execution_identity_sha256(base, census),
        producer.importance_execution_identity_sha256(fp16, census),
        producer.importance_execution_identity_sha256(
            changed_content, changed_census
        ),
        producer.importance_execution_identity_sha256(
            changed_producer, census
        ),
    }
    assert len(identities) == 4


def _payload(index: int, *, cover=None):
    cover = _cover() if cover is None else cover
    shard = cover["shards"][index]
    scale = index + 1
    dense = {
        "h_trace_raw": 16.0 * scale,
        "h_w2_sum_raw": 8.0 * scale,
        "h_trace": 2.0 * scale,
        "h_w2_sum": 1.0 * scale,
        "h_trace_norm_tokens": 8,
        "w_max_abs": 2.0,
        "w_norm_sq": 4.0,
        "n_params": 6,
        "in_features": 3,
        "out_features": 2,
        "n_tokens_seen": 8,
        "route_prob": None,
        "router_path": None,
        "expert_id": None,
        "fisher_row": np.array([4.0, 12.0], dtype=np.float32) * scale,
        "fisher_col": np.array([2.0, 6.0, 8.0], dtype=np.float32) * scale,
        "g_sq_sum": np.array([1.0, 2.0], dtype=np.float32) * scale,
        "act_sq_sum": np.array([3.0, 4.0, 5.0], dtype=np.float32) * scale,
        "act_absmax": np.array([2.0, 7.0 - index, 3.0], dtype=np.float32),
    }
    lm_head = copy.deepcopy(dense)
    lm_head["n_tokens_seen"] = 6
    mtp = copy.deepcopy(dense)
    mtp["n_tokens_seen"] = 4
    body_stats = {
        name: copy.deepcopy(dense)
        for name in cover["qname_census"]["activation_qname_manifest"][
            "entries"
        ]
    }
    return {
        "stats": {**body_stats, "lm_head": lm_head, "mtp.fc": mtp},
        "router_counts": {},
        "router_totals": {},
        "router_active_counts": {},
        "expert_route_stats": {},
        "expert_info": {},
        "meta": {
            "model": "model-content-id",
            "dataset": "test",
            "dtype": "bf16",
            "importance_weighting": False,
            "activation_rows_limit": 1024,
            "emit_marginals": True,
            "calibration_modality": "text-only",
            "packed_fisher_estimator": "per_token_v2",
            "sample_parallel_activation_scope": cover[
                "execution_identity"
            ]["activation_scope"],
            "sample_parallel_execution_identity": copy.deepcopy(
                cover["execution_identity"]
            ),
            "nsamples": 2,
            "seqlen": 4,
            "fisher_norm_tokens": 8,
            "calib_hash": shard["local_calibration_hash"],
            "activation_cache_dir": f"act-{index}",
            "sample_parallel": copy.deepcopy(shard),
        },
    }


def test_probe_merge_is_order_invariant_and_refinalizes_global_raw_stats():
    cover = _cover()
    shards = [_payload(0), _payload(1)]
    merged = merge_sample_parallel_probe_payloads(
        shards, expected_cover=cover
    )
    reversed_merge = merge_sample_parallel_probe_payloads(
        list(reversed(shards)), expected_cover=cover
    )

    assert pickle.dumps(merged) == pickle.dumps(reversed_merge)
    dense = merged["stats"][BODY_QNAMES[0]]
    assert dense["h_trace_raw"] == 48.0
    assert dense["h_trace"] == 3.0
    assert dense["h_trace_norm_tokens"] == 16
    np.testing.assert_array_equal(dense["act_absmax"], [2.0, 7.0, 3.0])
    assert merged["stats"]["lm_head"]["n_tokens_seen"] == 12
    assert merged["stats"]["mtp.fc"]["n_tokens_seen"] == 8
    assert merged["router_totals"] == {}
    provenance = merged["meta"]["sample_parallel_merge"]
    assert provenance["exact_disjoint_cover"] is True
    assert provenance["cover_identity_sha256"] == cover["identity_sha256"]


@pytest.mark.parametrize("mutation, match", [
    (
        lambda cover, payloads: cover["shards"][1].update(
            {"sample_start": 1}
        ),
        "range/provenance is invalid",
    ),
    (
        lambda cover, payloads: payloads[1]["stats"].pop("mtp.fc"),
        "qname cover differs",
    ),
    (
        lambda cover, payloads: payloads[1]["meta"].update(
            {"dtype": "fp16"}
        ),
        "execution meta 'dtype' differs",
    ),
    (
        lambda cover, payloads: payloads[0]["meta"].update(
            {"h_detail_dir": "/invalid-for-sample-merge"}
        ),
        "does not support h-detail",
    ),
])
def test_probe_merge_refuses_inexact_cover_or_payload_drift(mutation, match):
    cover = _cover()
    payloads = [_payload(0), _payload(1)]
    mutation(cover, payloads)
    with pytest.raises(SampleParallelMergeError, match=match):
        merge_sample_parallel_probe_payloads(
            payloads, expected_cover=cover
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("stamp.partition_index", True, "shard contract is invalid"),
        ("nsamples", True, "nsamples differs"),
        ("nsamples", "2", "nsamples differs"),
        ("seqlen", True, "seqlen differs"),
        ("seqlen", "4", "seqlen differs"),
        ("fisher_norm_tokens", True, "Fisher denominator differs"),
        ("fisher_norm_tokens", "8", "Fisher denominator differs"),
    ],
)
def test_probe_merge_refuses_bool_or_string_integer_contract_fields(
    field, value, match,
):
    cover = _cover()
    payloads = [_payload(0), _payload(1)]
    if field == "stamp.partition_index":
        payloads[0]["meta"]["sample_parallel"]["partition_index"] = value
    else:
        payloads[0]["meta"][field] = value
    with pytest.raises(SampleParallelMergeError, match=match):
        merge_sample_parallel_probe_payloads(payloads, expected_cover=cover)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda payload: payload["stats"]["mtp.fc"].update(
                {"n_tokens_seen": 8}
            ),
            "n_tokens_seen differs",
        ),
        (
            lambda payload: payload["stats"][BODY_QNAMES[0]].update(
                {"h_trace_raw": float("nan")}
            ),
            "finite and nonnegative",
        ),
        (
            lambda payload: payload["stats"]["mtp.fc"].update(
                {"expert_tokens": np.ones(2)}
            ),
            "closed dense schema",
        ),
        (
            lambda payload: payload.update(
                {"expert_info": {"mtp.fc": ["router", "0"]}}
            ),
            "routed/packed payload map",
        ),
    ],
)
def test_probe_merge_rejects_bad_raw_dense_or_routed_mtp(mutate, match):
    cover = _cover()
    payloads = [_payload(0), _payload(1)]
    mutate(payloads[1])
    with pytest.raises(SampleParallelMergeError, match=match):
        merge_sample_parallel_probe_payloads(payloads, expected_cover=cover)


@pytest.mark.parametrize("field", ["fisher_row", "fisher_col"])
def test_probe_merge_refuses_per_shard_fisher_marginal_trace_drift(field):
    cover = _cover()
    payloads = [_payload(0), _payload(1)]
    payloads[1]["stats"][BODY_QNAMES[0]][field][0] += np.float32(1.0)

    with pytest.raises(
        SampleParallelMergeError,
        match=rf"shard 1 sum\({field}\).*differs from h_trace_raw",
    ):
        merge_sample_parallel_probe_payloads(payloads, expected_cover=cover)


def test_probe_merge_rechecks_fisher_marginal_trace_after_addition(
    monkeypatch,
):
    original = merge_module.merge_marginals

    def corrupt_after_validated_input(dst, src):
        original(dst, src)
        if "fisher_row" in dst:
            dst["fisher_row"][0] += np.float32(1.0)

    monkeypatch.setattr(
        merge_module, "merge_marginals", corrupt_after_validated_input
    )
    with pytest.raises(
        SampleParallelMergeError,
        match=r"merged .* sum\(fisher_row\).*differs from h_trace_raw",
    ):
        merge_sample_parallel_probe_payloads(
            [_payload(0), _payload(1)], expected_cover=_cover()
        )


def test_probe_merge_fisher_marginal_tolerance_matches_sensitivity_card():
    payloads = [_payload(0), _payload(1)]
    trace = payloads[0]["stats"][BODY_QNAMES[0]]["h_trace_raw"]
    payloads[0]["stats"][BODY_QNAMES[0]]["fisher_row"][0] += np.float32(
        trace * 5e-4
    )

    # Different fp32 reduction orders are qualified to one part in 1,000,
    # matching SensitivityCard.validate's established contract.
    merge_sample_parallel_probe_payloads(payloads, expected_cover=_cover())


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda payload: payload["stats"][BODY_QNAMES[0]].update({
                "in_features": 2, "n_params": 4,
                "fisher_col": np.ones(2, dtype=np.float32),
                "act_sq_sum": np.ones(2, dtype=np.float32),
                "act_absmax": np.ones(2, dtype=np.float32),
            }),
            "geometry differs from the source",
        ),
        (
            lambda payload: payload["stats"][BODY_QNAMES[0]].update({
                "fisher_row": np.ones(2, dtype=np.float64),
            }),
            "dtype/shape differs from the source",
        ),
        (
            lambda payload: payload["stats"][BODY_QNAMES[0]].update({
                "packed_layout_v2": "invented",
            }),
            "closed dense schema",
        ),
    ],
)
def test_probe_merge_binds_geometry_and_closed_schema_even_when_all_shards_agree(
    mutate, match,
):
    cover = _cover()
    payloads = [_payload(0), _payload(1)]
    for payload in payloads:
        mutate(payload)
    with pytest.raises(SampleParallelMergeError, match=match):
        merge_sample_parallel_probe_payloads(payloads, expected_cover=cover)


def test_probe_merge_requires_every_top_level_audit_map_present():
    cover = _cover()
    payloads = [_payload(0), _payload(1)]
    for payload in payloads:
        payload.pop("router_active_counts")
    with pytest.raises(SampleParallelMergeError, match="producer payload fields"):
        merge_sample_parallel_probe_payloads(payloads, expected_cover=cover)


def test_cover_refuses_nonfirst_partition_provenance_drift():
    base = _cover()
    shards = copy.deepcopy(base["shards"])
    shards[1]["dataset"] = "other"
    with pytest.raises(SampleParallelMergeError, match="calibration partition"):
        build_sample_parallel_cover(
            shards,
            execution_identity=base["execution_identity"],
            qname_census=base["qname_census"],
        )


def test_importance_weighted_merge_requires_one_global_body_mean_receipt(tmp_path):
    base = _cover()
    execution = build_execution_identity(
        model="model-content-id", dataset="test", calib_seed=42,
        dtype="bf16", importance_weighting=True, emit_marginals=True,
        activation_rows_limit=1024, qname_census=base["qname_census"],
        **PRODUCER_IDENTITY,
    )
    cover = build_sample_parallel_cover(
        base["shards"], execution_identity=execution,
        qname_census=base["qname_census"],
    )
    payloads = [_payload(0), _payload(1)]
    for payload in payloads:
        payload["meta"]["importance_weighting"] = True
        payload["meta"]["sample_parallel_execution_identity"] = copy.deepcopy(
            execution
        )
    with pytest.raises(SampleParallelMergeError, match="global importance"):
        merge_sample_parallel_probe_payloads(
            payloads, expected_cover=cover
        )

    paths = []
    for index, shard in enumerate(cover["shards"]):
        path = tmp_path / f"local-{index}.json"
        write_local_importance_stats(
            path, partition_contract=shard,
            execution_identity_sha256=(
                producer.importance_execution_identity_sha256(
                    execution, base["qname_census"]
                )
            ),
            ce_sum=12.0 + 6.0 * index, ce_count=6,
        )
        paths.append(path)
    receipt = merge_importance_stats(paths, tmp_path / "global.json")
    for payload in payloads:
        payload["meta"]["sample_parallel_importance"] = copy.deepcopy(receipt)
    merged = merge_sample_parallel_probe_payloads(
        payloads, expected_cover=cover
    )
    assert merged["meta"]["sample_parallel_merge"][
        "importance_normalization_receipt"
    ] == receipt


def _write_activation(
    directory: Path,
    name: str,
    inputs: torch.Tensor,
    row_indices: torch.Tensor | None,
    *,
    partition_index: int,
    candidate_rows: int | None = None,
    cover=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    cover = _cover() if cover is None else cover
    shard = cover["shards"][partition_index]
    shard_tokens = (
        shard["sample_stop"] - shard["sample_start"]
    ) * cover["seqlen"]
    assert int(inputs.shape[0]) == shard_tokens
    row_indices = torch.arange(shard_tokens, dtype=torch.long)
    global_rows = row_indices + shard["sample_start"] * cover["seqlen"]
    priorities = activation_row_priorities(
        cover["global_calibration_hash"], name, global_rows
    )
    order = torch.argsort(global_rows, stable=True)
    order = order.index_select(
        0, torch.argsort(priorities.index_select(0, order), stable=True)
    )
    keep = order[:1024]
    inputs = inputs.index_select(0, keep)
    row_indices = row_indices.index_select(0, keep)
    priorities = priorities.index_select(0, keep)
    payload = {
        "name": name,
        "inputs": inputs,
        "row_indices": row_indices,
        "row_priorities": priorities,
        "sample_parallel_activation": {
            "schema": ACTIVATION_CACHE_SHARD_SCHEMA,
            "priority_schema": ACTIVATION_PRIORITY_SCHEMA,
            "priority_group": activation_priority_group(name),
            "selection": "local_top_r",
            "row_indices_scope": "shard_local_flat_tokens",
            "layout": "dense_linear",
            "global_calibration_hash": cover["global_calibration_hash"],
            "execution_identity_sha256": cover[
                "execution_identity"
            ]["identity_sha256"],
            "partition_index": partition_index,
            "rows_limit": 1024,
            "candidate_rows": (
                shard_tokens if candidate_rows is None else candidate_rows
            ),
        },
    }
    filename = name.replace(".", "__") + ".pt"
    torch.save(payload, directory / filename)


def test_activation_cache_merge_offsets_sorts_and_truncates_rows(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_inputs = torch.stack([
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32) + 0.5,
        torch.arange(8, dtype=torch.float32) + 1.0,
    ], dim=1)
    second_inputs = torch.stack([
        torch.arange(8, 16, dtype=torch.float32),
        torch.arange(8, 16, dtype=torch.float32) + 0.5,
        torch.arange(8, 16, dtype=torch.float32) + 1.0,
    ], dim=1)
    for qname in BODY_QNAMES:
        _write_activation(
            first, qname, first_inputs, None, partition_index=0,
        )
        _write_activation(
            second, qname, second_inputs, None, partition_index=1,
        )

    out_a = tmp_path / "merged-a"
    out_b = tmp_path / "merged-b"
    manifest_a = merge_sample_parallel_activation_caches(
        {0: first, 1: second},
        out_a,
        expected_cover=_cover(),
        max_rows=1024,
    )
    manifest_b = merge_sample_parallel_activation_caches(
        {1: second, 0: first},
        out_b,
        expected_cover=_cover(),
        max_rows=1024,
    )

    assert manifest_a == manifest_b
    assert (out_a / ACTIVATION_CACHE_MANIFEST).read_bytes() == (
        out_b / ACTIVATION_CACHE_MANIFEST
    ).read_bytes()
    dense = torch.load(
        out_a / (BODY_QNAMES[0].replace(".", "__") + ".pt"),
        map_location="cpu",
        weights_only=False,
    )
    candidate_rows = torch.arange(16, dtype=torch.long)
    candidate_priorities = activation_row_priorities(
        _cover()["global_calibration_hash"], BODY_QNAMES[0], candidate_rows
    )
    expected_order = torch.argsort(candidate_rows, stable=True)
    expected_order = expected_order.index_select(
        0,
        torch.argsort(
            candidate_priorities.index_select(0, expected_order), stable=True
        ),
    )[:1024]
    expected_inputs = torch.cat([first_inputs, second_inputs], dim=0)
    torch.testing.assert_close(
        dense["inputs"], expected_inputs.index_select(0, expected_order)
    )
    torch.testing.assert_close(
        dense["row_indices"], candidate_rows.index_select(0, expected_order)
    )
    assert manifest_a["activation_qname_manifest_sha256"] == _cover()[
        "qname_census"
    ]["activation_qname_manifest"]["identity_sha256"]
    with pytest.raises(SampleParallelMergeError, match="already exists"):
        merge_sample_parallel_activation_caches(
            {0: first, 1: second}, out_a, expected_cover=_cover(),
            max_rows=1024,
        )


def test_merge_bundle_is_atomic_after_cache_fault_and_exact_retry(
    monkeypatch, tmp_path,
):
    cover = _cover()
    caches = [tmp_path / "first", tmp_path / "second"]
    for index, cache in enumerate(caches):
        inputs = torch.arange(
            index * 24, (index + 1) * 24, dtype=torch.float32
        ).reshape(8, 3)
        _write_activation(
            cache, BODY_QNAMES[0], inputs, None,
            partition_index=index,
        )
    merged_probe = merge_sample_parallel_probe_payloads(
        [_payload(0), _payload(1)], expected_cover=cover,
    )
    destination = tmp_path / "bundle"

    monkeypatch.setenv(
        "PRISMAQUANT_TEST_FAULT_SAMPLE_MERGE_AFTER_CACHE", "1"
    )
    with pytest.raises(
        producer.SampleParallelProbeError, match="injected sample merge"
    ):
        producer.publish_sample_parallel_merge_bundle(
            merged_probe, {0: caches[0], 1: caches[1]}, destination,
            expected_cover=cover, max_rows=1024,
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".bundle.sample-merge-bundle-*")) == []

    monkeypatch.delenv(
        "PRISMAQUANT_TEST_FAULT_SAMPLE_MERGE_AFTER_CACHE"
    )
    commit = producer.publish_sample_parallel_merge_bundle(
        merged_probe, {0: caches[0], 1: caches[1]}, destination,
        expected_cover=cover, max_rows=1024,
    )
    assert {path.name for path in destination.iterdir()} == {
        producer.MERGE_BUNDLE_PROBE,
        producer.MERGE_BUNDLE_ACTIVATIONS,
        producer.MERGE_BUNDLE_COMMIT,
    }
    probe_bytes = (destination / producer.MERGE_BUNDLE_PROBE).read_bytes()
    assert commit["probe"]["sha256"] == hashlib.sha256(probe_bytes).hexdigest()
    persisted_commit = json.loads(
        (destination / producer.MERGE_BUNDLE_COMMIT).read_text()
    )
    assert persisted_commit == commit
    cache_manifest = json.loads(
        (
            destination / producer.MERGE_BUNDLE_ACTIVATIONS
            / ACTIVATION_CACHE_MANIFEST
        ).read_text()
    )
    assert commit["activation_cache"]["manifest_identity_sha256"] == (
        cache_manifest["identity_sha256"]
    )
    validated = producer.validate_sample_parallel_merge_bundle(destination)
    assert validated["identity_sha256"] == commit["identity_sha256"]
    from prismaquant.rtx4090_fp8_burn import _validate_sample_merge_bundle

    burn_validated = _validate_sample_merge_bundle(
        probe=destination / producer.MERGE_BUNDLE_PROBE,
        activation_cache_dir=(
            destination / producer.MERGE_BUNDLE_ACTIVATIONS
        ),
        commit_path=destination / producer.MERGE_BUNDLE_COMMIT,
    )
    assert burn_validated["source_model_content_sha256"] == validated[
        "source_model_content_sha256"
    ]
    assert burn_validated["source_model_upstream_content_sha256"] is None
    assert burn_validated[
        "source_model_upstream_portable_content_sha256"
    ] is None
    with pytest.raises(
        producer.SampleParallelProbeError, match="already exists"
    ):
        producer.publish_sample_parallel_merge_bundle(
            merged_probe, {0: caches[0], 1: caches[1]}, destination,
            expected_cover=cover, max_rows=1024,
        )

    # Even a self-consistently re-sealed commit cannot authorize altered
    # globally derived Fisher normalization.
    probe_path = destination / producer.MERGE_BUNDLE_PROBE
    with probe_path.open("rb") as handle:
        altered_probe = pickle.load(handle)
    altered_probe["stats"][BODY_QNAMES[0]]["h_trace"] += 1.0
    altered_bytes = pickle.dumps(
        altered_probe, protocol=pickle.HIGHEST_PROTOCOL
    )
    probe_path.write_bytes(altered_bytes)
    commit["probe"]["bytes"] = len(altered_bytes)
    commit["probe"]["sha256"] = hashlib.sha256(altered_bytes).hexdigest()
    commit_body = dict(commit)
    commit_body.pop("identity_sha256")
    commit["identity_sha256"] = producer._canonical_sha256(
        commit_body, where="re-sealed altered bundle commit"
    )
    (destination / producer.MERGE_BUNDLE_COMMIT).write_text(
        json.dumps(commit)
    )
    with pytest.raises(
        producer.SampleParallelProbeError, match="normalization differs"
    ):
        producer.validate_sample_parallel_merge_bundle(destination)


@pytest.mark.parametrize(
    "siblings",
    [
        ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
         "model.layers.0.self_attn.v_proj"),
        ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"),
    ],
)
def test_activation_priorities_align_fused_sibling_rows(siblings):
    rows = torch.arange(32, dtype=torch.long)
    priorities = [
        activation_row_priorities("a" * 32, name, rows)
        for name in siblings
    ]
    assert len({activation_priority_group(name) for name in siblings}) == 1
    for observed in priorities[1:]:
        torch.testing.assert_close(observed, priorities[0])


def test_activation_cache_merge_refuses_missing_exact_priority_provenance(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_activation(
        first, BODY_QNAMES[0], torch.ones(8, 3), None,
        partition_index=0,
    )
    _write_activation(
        second, BODY_QNAMES[0], torch.ones(8, 3), None,
        partition_index=1,
    )
    second_file = second / (BODY_QNAMES[0].replace(".", "__") + ".pt")
    blob = torch.load(second_file, map_location="cpu", weights_only=False)
    blob.pop("row_priorities")
    torch.save(blob, second_file)
    with pytest.raises(
        SampleParallelMergeError, match="malformed|invalid priority"
    ):
        merge_sample_parallel_activation_caches(
            {0: first, 1: second},
            tmp_path / "merged",
            expected_cover=_cover(),
            max_rows=1024,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "packed_blob_marker", "expert_stamp_marker", "float_row_indices",
        "string_counts", "bool_partition",
    ],
)
def test_activation_cache_merge_refuses_open_blob_or_coerced_indices(
    tmp_path, mutation,
):
    caches = [tmp_path / "first", tmp_path / "second"]
    for index, cache in enumerate(caches):
        _write_activation(
            cache, BODY_QNAMES[0], torch.ones(8, 3), None,
            partition_index=index,
        )
    path = caches[1] / (BODY_QNAMES[0].replace(".", "__") + ".pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if mutation == "packed_blob_marker":
        blob["packed_experts"] = True
    elif mutation == "expert_stamp_marker":
        blob["sample_parallel_activation"]["expert_payload"] = []
    elif mutation == "float_row_indices":
        blob["row_indices"] = blob["row_indices"].to(torch.float32)
    elif mutation == "string_counts":
        blob["sample_parallel_activation"]["rows_limit"] = "1024"
        blob["sample_parallel_activation"]["candidate_rows"] = "8"
    else:
        blob["sample_parallel_activation"]["partition_index"] = True
    torch.save(blob, path)
    with pytest.raises(
        SampleParallelMergeError, match="malformed|invalid priority"
    ):
        merge_sample_parallel_activation_caches(
            {0: caches[0], 1: caches[1]}, tmp_path / "merged-closed",
            expected_cover=_cover(), max_rows=1024,
        )


def test_activation_cache_merge_refuses_pre_group_priority_schema(tmp_path):
    caches = [tmp_path / "first", tmp_path / "second"]
    for index, cache in enumerate(caches):
        _write_activation(
            cache, BODY_QNAMES[0], torch.ones(8, 3), None,
            partition_index=index,
        )
    path = caches[1] / (BODY_QNAMES[0].replace(".", "__") + ".pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["sample_parallel_activation"]["priority_schema"] = (
        "blake2b63-global-row-v1"
    )
    torch.save(blob, path)
    with pytest.raises(SampleParallelMergeError, match="invalid priority"):
        merge_sample_parallel_activation_caches(
            {0: caches[0], 1: caches[1]}, tmp_path / "merged",
            expected_cover=_cover(),
            max_rows=1024,
        )


@pytest.mark.parametrize(
    "inputs,match",
    [
        (torch.ones(8, 2, dtype=torch.float32), "dtype/width"),
        (torch.ones(8, 3, dtype=torch.bfloat16), "dtype/width"),
    ],
)
def test_activation_merge_binds_source_width_and_fp32_cache_dtype(
    tmp_path, inputs, match,
):
    caches = [tmp_path / "first", tmp_path / "second"]
    for index, cache in enumerate(caches):
        _write_activation(
            cache, BODY_QNAMES[0], inputs.clone(), None,
            partition_index=index,
        )
    with pytest.raises(SampleParallelMergeError, match=match):
        merge_sample_parallel_activation_caches(
            {0: caches[0], 1: caches[1]},
            tmp_path / "merged-geometry",
            expected_cover=_cover(),
            max_rows=1024,
        )


def test_merger_priority_verifier_is_independent_of_producer_torch_helper(
    monkeypatch, tmp_path,
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    second_inputs = torch.arange(24, 48, dtype=torch.float32).reshape(8, 3)
    _write_activation(
        first, BODY_QNAMES[0], first_inputs, None, partition_index=0,
    )
    _write_activation(
        second, BODY_QNAMES[0], second_inputs, None, partition_index=1,
    )
    monkeypatch.setattr(
        merge_module,
        "activation_row_priorities",
        lambda _global_hash, _qname, rows: torch.zeros_like(rows),
    )
    output = tmp_path / "merged-independent"
    manifest = merge_sample_parallel_activation_caches(
        {1: second, 0: first},
        output,
        expected_cover=_cover(),
        max_rows=1024,
    )
    assert manifest["independent_numpy_priority_verifier"] is True
    assert manifest["local_global_top_r_associativity_verified"] is True


def test_independent_scalar_numpy_and_torch_priority_parity_and_associativity():
    calibration_hash = "a" * 32
    qname = "model.layers.0.self_attn.q_proj"
    rows = np.array(
        [0, 1, 65535, 65536, (1 << 31) - 1, 1 << 31, (1 << 32) - 1],
        dtype=np.int64,
    )
    numpy_priorities = merge_module._reference_activation_priorities_numpy(
        calibration_hash, qname, rows
    )
    scalar_priorities = np.array([
        merge_module._reference_activation_priority_scalar(
            calibration_hash, qname, int(row)
        )
        for row in rows
    ], dtype=np.int64)
    torch_priorities = activation_row_priorities(
        calibration_hash, qname, torch.from_numpy(rows.copy())
    ).numpy()
    np.testing.assert_array_equal(numpy_priorities, scalar_priorities)
    np.testing.assert_array_equal(numpy_priorities, torch_priorities)

    all_rows = np.arange(1024, dtype=np.int64)
    monolithic_rows, monolithic_priorities = (
        merge_module._reference_top_r_numpy(
            calibration_hash, qname, all_rows, 37
        )
    )
    local_plans = [
        merge_module._reference_top_r_numpy(
            calibration_hash, qname, partition, 37
        )
        for partition in np.array_split(all_rows, 7)
    ]
    union_rows = np.concatenate([plan[0] for plan in local_plans])
    associated_rows, associated_priorities = (
        merge_module._reference_top_r_numpy(
            calibration_hash, qname, union_rows, 37
        )
    )
    np.testing.assert_array_equal(associated_rows, monolithic_rows)
    np.testing.assert_array_equal(
        associated_priorities, monolithic_priorities
    )
    replayed_rows, replayed_priorities = (
        merge_module._reference_top_r_global_domain_numpy(
            calibration_hash, qname, 1024, 37, chunk_rows=113
        )
    )
    np.testing.assert_array_equal(replayed_rows, monolithic_rows)
    np.testing.assert_array_equal(replayed_priorities, monolithic_priorities)


def _publish_valid_activation_bundle(
    tmp_path: Path,
    *,
    body_qnames: tuple[str, ...] = BODY_QNAMES,
):
    cover = _cover(body_qnames=body_qnames)
    caches = [tmp_path / "first", tmp_path / "second"]
    for index, cache in enumerate(caches):
        inputs = torch.arange(
            index * 24, (index + 1) * 24, dtype=torch.float32
        ).reshape(8, 3)
        for qname in body_qnames:
            _write_activation(
                cache,
                qname,
                inputs.clone(),
                None,
                partition_index=index,
                cover=cover,
            )
    merged_probe = merge_sample_parallel_probe_payloads(
        [_payload(0, cover=cover), _payload(1, cover=cover)],
        expected_cover=cover,
    )
    destination = tmp_path / "bundle"
    producer.publish_sample_parallel_merge_bundle(
        merged_probe,
        {0: caches[0], 1: caches[1]},
        destination,
        expected_cover=cover,
        max_rows=1024,
    )
    validate_merged_activation_cache_output(
        destination / producer.MERGE_BUNDLE_ACTIVATIONS,
        expected_cover=cover,
    )
    producer.validate_sample_parallel_merge_bundle(
        destination, expected_cover=cover
    )
    return cover, destination


def test_burn_col_weights_use_captured_validated_bundle_probe_after_replace(
    monkeypatch, tmp_path,
):
    from argparse import Namespace

    from prismaquant.cb_imatrix import (
        canonical_imatrix_sha256,
        imatrix_from_probe_stats,
    )
    from prismaquant.rtx4090_fp8_burn import derive_col_weights

    _cover_value, bundle = _publish_valid_activation_bundle(tmp_path)
    real_validate = producer.validate_sample_parallel_merge_bundle
    captured = {}

    def _validate_then_replace(bundle_dir, *, capture_consumables=False):
        validated = real_validate(
            bundle_dir, capture_consumables=capture_consumables,
        )
        captured.update(validated)
        replacement_payload = copy.deepcopy(
            validated["_validated_probe_payload"]
        )
        replacement_payload["stats"][BODY_QNAMES[0]][
            "act_sq_sum"
        ] = np.full(3, 999.0, dtype=np.float32)
        replacement = tmp_path / "post-validation-replacement.pkl"
        replacement.write_bytes(pickle.dumps(
            replacement_payload, protocol=pickle.HIGHEST_PROTOCOL,
        ))
        replacement.replace(bundle / producer.MERGE_BUNDLE_PROBE)
        return validated

    monkeypatch.setattr(
        producer, "validate_sample_parallel_merge_bundle",
        _validate_then_replace,
    )
    output = tmp_path / "cb_col_weights.pkl"
    assert derive_col_weights(Namespace(
        sample_merge_bundle=str(bundle), output=str(output),
    )) == output

    expected, _provenance = imatrix_from_probe_stats(
        captured["_validated_probe_payload"]["stats"]
    )
    with output.open("rb") as handle:
        observed = pickle.load(handle)
    assert isinstance(observed, dict)
    assert canonical_imatrix_sha256(observed) == canonical_imatrix_sha256(
        expected
    )
    assert set(observed) == set(expected)
    for qname in expected:
        assert observed[qname].dtype == torch.float32
        torch.testing.assert_close(observed[qname], expected[qname])


def test_burn_col_weight_derivation_rejects_invalid_bundle_without_output(
    tmp_path,
):
    from argparse import Namespace

    from prismaquant.rtx4090_fp8_burn import (
        RTX4090FP8BurnError,
        derive_col_weights,
    )

    _cover_value, bundle = _publish_valid_activation_bundle(tmp_path)
    (bundle / producer.MERGE_BUNDLE_COMMIT).write_text("{}\n")
    output = tmp_path / "cb_col_weights.pkl"
    with pytest.raises(
        RTX4090FP8BurnError, match="bundle validation failed",
    ):
        derive_col_weights(Namespace(
            sample_merge_bundle=str(bundle), output=str(output),
        ))
    assert not output.exists()


def _reseal_activation_payload_and_bundle_commit(
    bundle: Path,
    qname: str,
    payload: dict[str, object],
):
    activation_dir = bundle / producer.MERGE_BUNDLE_ACTIVATIONS
    payload_path = activation_dir / (qname.replace(".", "__") + ".pt")
    identity = {
        "inputs": merge_module._tensor_identity(payload["inputs"]),
        "row_indices": merge_module._tensor_identity(payload["row_indices"]),
        "row_priorities": merge_module._tensor_identity(
            payload["row_priorities"]
        ),
        "source_shards": [0, 1],
        "priority_group": activation_priority_group(qname),
    }
    identity["identity_sha256"] = merge_module.canonical_json_sha256(
        identity, where=f"re-sealed adversarial activation {qname}"
    )
    payload["sample_parallel_merge"]["identity_sha256"] = identity[
        "identity_sha256"
    ]
    torch.save(payload, payload_path)

    manifest_path = activation_dir / ACTIVATION_CACHE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][qname] = identity
    manifest_body = dict(manifest)
    manifest_body.pop("identity_sha256")
    manifest["identity_sha256"] = merge_module.canonical_json_sha256(
        manifest_body, where="re-sealed adversarial activation manifest"
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )

    commit_path = bundle / producer.MERGE_BUNDLE_COMMIT
    commit = json.loads(commit_path.read_text())
    commit["activation_cache"]["manifest_identity_sha256"] = manifest[
        "identity_sha256"
    ]
    commit_body = dict(commit)
    commit_body.pop("identity_sha256")
    commit["identity_sha256"] = producer._canonical_sha256(
        commit_body, where="re-sealed adversarial bundle commit"
    )
    commit_path.write_text(
        json.dumps(commit, sort_keys=True, indent=2) + "\n"
    )


@pytest.mark.parametrize(
    "mutation, expected_match",
    [
        ("reversed", "canonical exact global top-R"),
        ("duplicate", "not unique"),
        ("wrong_domain", "global row domain"),
        ("wrong_priority", "priorities differ"),
        ("fused_misaligned", "fused siblings"),
        ("short_cardinality", "tensor contract"),
    ],
)
def test_public_and_bundle_validators_refuse_resealed_activation_semantics(
    tmp_path, mutation, expected_match,
):
    fused_qnames = (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
    )
    body_qnames = fused_qnames if mutation == "fused_misaligned" else BODY_QNAMES
    cover, bundle = _publish_valid_activation_bundle(
        tmp_path, body_qnames=body_qnames
    )
    qname = fused_qnames[0] if mutation == "fused_misaligned" else BODY_QNAMES[0]
    payload_path = (
        bundle
        / producer.MERGE_BUNDLE_ACTIVATIONS
        / (qname.replace(".", "__") + ".pt")
    )
    payload = torch.load(
        payload_path, map_location="cpu", weights_only=False
    )
    inputs = payload["inputs"].clone()
    rows = payload["row_indices"].clone()
    priorities = payload["row_priorities"].clone()
    if mutation in {"reversed", "fused_misaligned"}:
        order = torch.arange(rows.numel() - 1, -1, -1, dtype=torch.long)
        inputs = inputs.index_select(0, order)
        rows = rows.index_select(0, order)
        priorities = priorities.index_select(0, order)
    elif mutation == "duplicate":
        inputs[1] = inputs[0]
        rows[1] = rows[0]
        priorities[1] = priorities[0]
    elif mutation == "wrong_domain":
        rows[0] = int(cover["total_samples"]) * int(cover["seqlen"])
        priorities = activation_row_priorities(
            cover["global_calibration_hash"], qname, rows
        )
    elif mutation == "wrong_priority":
        priorities[0] = torch.bitwise_xor(
            priorities[0], torch.ones((), dtype=torch.int64)
        )
    elif mutation == "short_cardinality":
        inputs = inputs[:-1].contiguous()
        rows = rows[:-1].contiguous()
        priorities = priorities[:-1].contiguous()
    else:  # pragma: no cover - closes the adversarial table
        raise AssertionError(mutation)
    payload["inputs"] = inputs
    payload["row_indices"] = rows
    payload["row_priorities"] = priorities
    _reseal_activation_payload_and_bundle_commit(
        bundle, qname, payload
    )

    with pytest.raises(SampleParallelMergeError, match=expected_match):
        validate_merged_activation_cache_output(
            bundle / producer.MERGE_BUNDLE_ACTIVATIONS,
            expected_cover=cover,
        )
    with pytest.raises(
        producer.SampleParallelProbeError, match=expected_match
    ):
        producer.validate_sample_parallel_merge_bundle(
            bundle, expected_cover=cover
        )


def test_public_and_bundle_validators_refuse_resealed_manifest_max_rows(
    tmp_path,
):
    cover, bundle = _publish_valid_activation_bundle(tmp_path)
    activation_dir = bundle / producer.MERGE_BUNDLE_ACTIVATIONS
    manifest_path = activation_dir / ACTIVATION_CACHE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["max_rows"] = 16
    manifest_body = dict(manifest)
    manifest_body.pop("identity_sha256")
    manifest["identity_sha256"] = merge_module.canonical_json_sha256(
        manifest_body, where="re-sealed wrong max_rows manifest"
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    commit_path = bundle / producer.MERGE_BUNDLE_COMMIT
    commit = json.loads(commit_path.read_text())
    commit["activation_cache"]["manifest_identity_sha256"] = manifest[
        "identity_sha256"
    ]
    commit_body = dict(commit)
    commit_body.pop("identity_sha256")
    commit["identity_sha256"] = producer._canonical_sha256(
        commit_body, where="re-sealed wrong max_rows bundle commit"
    )
    commit_path.write_text(
        json.dumps(commit, sort_keys=True, indent=2) + "\n"
    )

    with pytest.raises(SampleParallelMergeError, match="identity differs"):
        validate_merged_activation_cache_output(
            activation_dir, expected_cover=cover
        )
    with pytest.raises(
        producer.SampleParallelProbeError, match="identity differs"
    ):
        producer.validate_sample_parallel_merge_bundle(
            bundle, expected_cover=cover
        )


def test_bundle_probe_replace_between_capture_and_deserialize_is_refused(
    monkeypatch, tmp_path,
):
    cover, bundle = _publish_valid_activation_bundle(tmp_path)
    probe_path = bundle / producer.MERGE_BUNDLE_PROBE
    original_loads = pickle.loads
    replaced = False

    def _replace_after_capture(payload):
        nonlocal replaced
        if not replaced:
            replacement = bundle / "replacement-probe.pkl"
            replacement.write_bytes(probe_path.read_bytes())
            replacement.replace(probe_path)
            replaced = True
        return original_loads(payload)

    monkeypatch.setattr(pickle, "loads", _replace_after_capture)
    with pytest.raises(
        producer.SampleParallelProbeError,
        match="probe changed during validation",
    ):
        producer.validate_sample_parallel_merge_bundle(
            bundle, expected_cover=cover, capture_consumables=True,
        )


def test_verified_activation_index_refuses_post_validation_replace(tmp_path):
    from prismaquant.measure_quant_cost import ActivationIndex

    cover, bundle = _publish_valid_activation_bundle(tmp_path)
    validated = producer.validate_sample_parallel_merge_bundle(
        bundle, expected_cover=cover, capture_consumables=True,
    )
    assert validated["_validated_probe_bytes"] == (
        bundle / producer.MERGE_BUNDLE_PROBE
    ).read_bytes()
    qname = BODY_QNAMES[0]
    activation_dir = bundle / producer.MERGE_BUNDLE_ACTIVATIONS
    index = ActivationIndex(
        activation_dir,
        {qname: {}},
        verification_contract=validated[
            "_validated_activation_manifest"
        ],
    )
    try:
        payload_path = activation_dir / (qname.replace(".", "__") + ".pt")
        payload = torch.load(
            payload_path, map_location="cpu", weights_only=False
        )
        payload["inputs"] = payload["inputs"].clone()
        payload["inputs"][0, 0] += 1.0
        replacement = activation_dir / "replacement.pt"
        torch.save(payload, replacement)
        replacement.replace(payload_path)
        with pytest.raises(ValueError, match="differs from commit"):
            index.load_blob(qname)
    finally:
        index.close()
