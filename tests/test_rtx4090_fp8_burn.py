from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant.rtx4090_fp8_burn import (
    ALLOCATOR_PROBE_FD,
    AURA_N_PROBES,
    AURA_TOKEN_SCOPE,
    BF16_FORMAT,
    CB_COMPILE_PROOF_KEY,
    CB_FORMATS,
    EXECUTION_ATTESTATION_SCHEMA,
    FULL_FORMATS,
    IMATRIX_CONTRACT_SCHEMA,
    MEASURED_FORMATS,
    MEASURED_CB_FORMATS,
    NATIVE_FP8_FORMAT,
    PLAN_SCHEMA,
    RENDER_FORMATS,
    RTX4090FP8BurnError,
    SHARD_RECEIPT_KEY,
    SHARD_RECEIPT_SCHEMA,
    SOURCE_IDENTITY_BINDING_SCHEMA,
    STREAMING_CACHE_MAX_SLOTS,
    STREAMING_PREFETCH_LOOKAHEAD,
    STREAMING_REQUIRE_PREFETCHED_RESIDENCY,
    _allocator_cost,
    _arm_identity,
    _attach_campaign_shard_receipt,
    _authoritative_linear_source_census,
    _build_parser,
    _cb_context,
    _calibration_contract,
    _probe_imatrix_contract,
    _require_compile_settings,
    _revalidate_live_campaign_census,
    _source_identity_binding,
    _validate_execution_attestation,
    _validate_campaign_shards,
    _validate_merged_campaign_provenance,
    _validate_live_sample_source_census,
    _validate_sample_bundle_source_binding,
    _verify_source_identity_binding,
    allocate,
    attest_execution,
    build_execution_attestation,
    build_campaign_plan,
    derive_col_weights,
    measure,
    validate_campaign_plan,
)
from prismaquant.cb_compile_contract import (
    CB_COMPILE_FAIL_CLOSED_ENV,
    begin_cb_compile_execution_proof,
    finish_cb_compile_execution_proof,
)
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.production_weight_cache import (
    bind_cb_render_identity_source_weights,
    build_production_cache_cb_render_identity,
)
from prismaquant.rtx4090_cb_compile_proof import (
    AURA_CHECKPOINT_BINDING_SCHEMA,
    AURA_CHECKPOINT_IDENTITY_SCHEMA,
    AURA_CHECKPOINT_MANIFEST_SCHEMA,
    build_campaign_cb_compile_proof,
)
from prismaquant.sample_parallel_probe import (
    BODY_STATS_AND_ACTIVATION,
    LM_HEAD_STATS_ONLY,
    MTP_STATS_ONLY,
)


class _Profile:
    def fused_sibling_group(self, _qname):
        return None

    def packed_expert_format_group(self, _qname):
        return None

    def is_pinned_name(self, qname):
        return qname == "lm_head"


def _sample_execution_identity():
    return {
        "identity_sha256": "1" * 64,
        "producer_snapshot_sha256": "2" * 64,
        "producer_snapshot_commit": "3" * 40,
        "producer_snapshot_tree": "4" * 40,
        "container_image_digest": "sha256:" + "5" * 64,
    }


def _producer_snapshot_identity():
    return {
        "closure_sha256": "2" * 64,
        "commit": "3" * 40,
        "tree": "4" * 40,
    }


def test_closed_execution_attestation_matching_contract_is_accepted(tmp_path):
    execution = _sample_execution_identity()
    snapshot = _producer_snapshot_identity()
    payload = build_execution_attestation(
        execution,
        producer_snapshot=snapshot,
        launcher_image_digest=execution["container_image_digest"],
    )
    path = tmp_path / "execution-attestation.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    assert _validate_execution_attestation(
        path,
        execution_identity=execution,
        producer_snapshot=snapshot,
        launcher_image_digest=execution["container_image_digest"],
    ) == payload


@pytest.mark.parametrize("tamper", ["malformed", "resealed"])
def test_closed_execution_attestation_refuses_malformed_or_resealed(
    tamper, tmp_path,
):
    execution = _sample_execution_identity()
    snapshot = _producer_snapshot_identity()
    payload = build_execution_attestation(
        execution,
        producer_snapshot=snapshot,
        launcher_image_digest=execution["container_image_digest"],
    )
    if tamper == "malformed":
        payload["identity_sha256"] = "0" * 64
    else:
        payload["container_image_digest"] = "sha256:" + "6" * 64
        body = dict(payload)
        body.pop("identity_sha256")
        payload["identity_sha256"] = canonical_json_sha256(
            body, where="resealed execution attestation",
        )
    path = tmp_path / "execution-attestation.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(
        RTX4090FP8BurnError,
        match=("checksum differs" if tamper == "malformed" else "differs"),
    ):
        _validate_execution_attestation(
            path,
            execution_identity=execution,
            producer_snapshot=snapshot,
            launcher_image_digest=execution["container_image_digest"],
        )


def test_second_stripe_different_launcher_image_is_refused_before_cuda(
    tmp_path,
):
    execution = _sample_execution_identity()
    snapshot = _producer_snapshot_identity()
    payload = build_execution_attestation(
        execution,
        producer_snapshot=snapshot,
        launcher_image_digest=execution["container_image_digest"],
    )
    path = tmp_path / "execution-attestation.json"
    path.write_text(json.dumps(payload, sort_keys=True))

    # Stripe 0's trusted launcher value agrees with the common contract.
    _validate_execution_attestation(
        path,
        execution_identity=execution,
        producer_snapshot=snapshot,
        launcher_image_digest=execution["container_image_digest"],
    )
    # This is the same validator measure() calls before GPU imports/CUDA.
    with pytest.raises(RTX4090FP8BurnError, match="trusted launcher image"):
        _validate_execution_attestation(
            path,
            execution_identity=execution,
            producer_snapshot=snapshot,
            launcher_image_digest="sha256:" + "6" * 64,
        )


def _attestation_publication_args(monkeypatch, tmp_path, output):
    import prismaquant.rtx4090_fp8_burn as burn
    import prismaquant.sample_parallel_probe as sample_probe

    execution = _sample_execution_identity()
    snapshot = _producer_snapshot_identity()
    run_contract = tmp_path / "run-contract.json"
    run_contract.write_text("{}")
    monkeypatch.setattr(
        burn, "_validate_burn_runtime_snapshot", lambda _path: snapshot,
    )
    monkeypatch.setattr(
        sample_probe,
        "validate_run_contract",
        lambda _raw: {"execution_identity": execution},
    )
    return SimpleNamespace(
        sample_run_contract=str(run_contract),
        producer_snapshot=str(tmp_path / "snapshot.json"),
        launcher_image_digest=execution["container_image_digest"],
        output=str(output),
    )


def test_execution_attestation_race_preserves_competing_file(
    monkeypatch, tmp_path,
):
    import prismaquant.sample_parallel_probe as sample_probe

    output = tmp_path / "execution-attestation.json"
    args = _attestation_publication_args(monkeypatch, tmp_path, output)
    real_link = sample_probe.os.link

    def _race_link(source, destination):
        assert Path(destination) == output
        output.write_bytes(b"competing-attestation")
        return real_link(source, destination)

    monkeypatch.setattr(sample_probe.os, "link", _race_link)
    with pytest.raises(
        RTX4090FP8BurnError, match="publication failed.*refusing to overwrite",
    ):
        attest_execution(args)
    assert output.read_bytes() == b"competing-attestation"


def test_execution_attestation_refuses_and_preserves_dangling_symlink(
    monkeypatch, tmp_path,
):
    output = tmp_path / "execution-attestation.json"
    target = tmp_path / "missing-attestation-target"
    output.symlink_to(target)
    args = _attestation_publication_args(monkeypatch, tmp_path, output)
    with pytest.raises(
        RTX4090FP8BurnError, match="publication failed.*refusing to overwrite",
    ):
        attest_execution(args)
    assert output.is_symlink()
    assert os.readlink(output) == str(target)


def test_measure_refuses_different_launcher_image_before_cuda_request(
    monkeypatch, tmp_path,
):
    import prismaquant.gpu_guard as gpu_guard
    import prismaquant.rtx4090_fp8_burn as burn

    execution = _sample_execution_identity()
    snapshot = _producer_snapshot_identity()
    payload = build_execution_attestation(
        execution,
        producer_snapshot=snapshot,
        launcher_image_digest=execution["container_image_digest"],
    )
    attestation = tmp_path / "execution-attestation.json"
    attestation.write_text(json.dumps(payload, sort_keys=True))
    monkeypatch.setattr(burn, "load_campaign_plan", lambda _path: _plan())
    monkeypatch.setattr(
        burn, "_validate_burn_runtime_snapshot", lambda _path: snapshot,
    )
    monkeypatch.setattr(
        burn,
        "_require_compile_settings",
        lambda: {
            "PRISMAQUANT_CB_ENCODE_COMPILE": "1",
            "PRISMAQUANT_CB_ATOM_COMPILE": "1",
            CB_COMPILE_FAIL_CLOSED_ENV: "1",
        },
    )
    monkeypatch.setattr(burn, "_verify_binding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        burn,
        "_verify_source_identity_binding",
        lambda *_args, **_kwargs: {"content_sha256": "4" * 64},
    )
    monkeypatch.setattr(
        burn,
        "_validate_sample_merge_bundle",
        lambda **_kwargs: {"_validated_execution_identity": execution},
    )
    monkeypatch.setattr(
        gpu_guard,
        "require_cuda_hot_path",
        lambda *_args, **_kwargs: pytest.fail("CUDA was requested"),
    )
    args = SimpleNamespace(
        plan="plan.json", producer_snapshot="snapshot.json", stripe=1,
        n_probes=AURA_N_PROBES, probe="probe.pkl",
        col_weights="col-weights.pkl",
        execution_attestation=str(attestation), dataset="dataset.jsonl",
        sample_merge_commit="commit.json",
        activation_cache_dir="activation-cache",
        model="model", source_identity="source-identity.json",
        launcher_image_digest="sha256:" + "6" * 64,
    )
    with pytest.raises(RTX4090FP8BurnError, match="trusted launcher image"):
        measure(args)


def _test_streamed_identity(
    *,
    root: str = "/model",
    staged_from: str = "/tmp/prismaquant_stage_a",
    semantic_tag: str = "qualified-source",
):
    config = {
        "model_type": "qwen3_5",
        "hidden_size": 8,
        "semantic_tag": semantic_tag,
        "_name_or_path": staged_from,
        "transformers_version": "test-runtime",
    }
    weight_map = {
        "model.layers.0.proj.weight": "model.layers.0.proj.weight",
    }
    checkpoint_weight_map = {
        "model.layers.0.proj.weight": "model.safetensors",
    }
    shards = [{
        "path": str(Path(root) / "model.safetensors"),
        "size": 1,
        "sha256": "c" * 64,
    }]
    value_bearing = {
        "config": config,
        "weight_map": weight_map,
        "shards": shards,
        "checkpoint_weight_map": checkpoint_weight_map,
    }
    return {
        "schema": "prismaquant.streamed_model.identity.v1",
        "source": root,
        "resolved_commit": None,
        "content_sha256": canonical_json_sha256(
            value_bearing, where="test streamed model content",
        ),
        **value_bearing,
    }


def _test_portable_content(identity):
    from prismaquant.cost_streaming import (
        portable_streamed_model_content_identity,
    )

    return portable_streamed_model_content_identity(
        identity, where="test portable streamed model content",
    )["portable_content_sha256"]


def test_sample_bundle_source_bridge_uses_upstream_not_portable_digest():
    live = _test_streamed_identity()
    upstream = _test_portable_content(live)
    bundle = {
        "source_model_content_sha256": "a" * 64,
        "source_model_upstream_portable_content_sha256": upstream,
    }
    binding = {"portable_content_sha256": upstream}
    assert _validate_sample_bundle_source_binding(
        bundle, binding, live_source_identity=live,
    ) == {
        "portable_content_sha256": "a" * 64,
        "upstream_portable_content_sha256": upstream,
    }


def test_host_local_source_caches_share_one_value_bearing_plan_binding(
    monkeypatch, tmp_path,
):
    import prismaquant.cost_streaming as cost_streaming

    identity_a = _test_streamed_identity(
        root="/sparky/model", staged_from="/tmp/prismaquant_stage_sparky",
    )
    identity_b = _test_streamed_identity(
        root="/sparklina/model",
        staged_from="/tmp/prismaquant_stage_sparklina",
    )
    assert identity_a["content_sha256"] != identity_b["content_sha256"]
    assert _test_portable_content(identity_a) == _test_portable_content(identity_b)
    cache_a = tmp_path / "source-identity-cache-a.json"
    cache_b = tmp_path / "source-identity-cache-b.json"
    cache_a.write_text('{"host":"sparky","inode":1}')
    cache_b.write_text('{"host":"sparklina","inode":999}')
    identities = {
        str(cache_a): dict(identity_a),
        str(cache_b): dict(identity_b),
    }

    def _validate(_model, path, *, require_complete_checkpoint):
        assert require_complete_checkpoint is True
        return dict(identities[str(path)])

    monkeypatch.setattr(
        cost_streaming,
        "validate_cached_streamed_model_identity",
        _validate,
    )
    prepared, live = _source_identity_binding("model", cache_a)
    assert prepared["schema"] == SOURCE_IDENTITY_BINDING_SCHEMA
    assert prepared["portable_content_sha256"] == _test_portable_content(
        identity_a
    )
    assert live == identity_a
    plan = {"bindings": {"source_model_identity": prepared}}
    assert _verify_source_identity_binding(
        plan, model="model", cache_path=cache_b,
    ) == identity_b

    identities[str(cache_b)] = _test_streamed_identity(
        root="/sparklina/model",
        staged_from="/tmp/prismaquant_stage_sparklina",
        semantic_tag="different-model",
    )
    with pytest.raises(
        RTX4090FP8BurnError, match="differs from the prepared plan",
    ):
        _verify_source_identity_binding(
            plan, model="model", cache_path=cache_b,
        )


@pytest.mark.parametrize("changed", ["portable", "projection"])
def test_live_portable_source_census_must_match_captured_bundle(
    monkeypatch, changed,
):
    import prismaquant.sample_parallel_probe as sample_probe

    live_census = {
        "source_census": {
            "source_model_identity": {
                "content_sha256": "a" * 64,
                "upstream_portable_content_sha256": "d" * 64,
            },
        },
        "stable_projection": {"identity_sha256": "b" * 64},
    }
    monkeypatch.setattr(
        sample_probe, "build_rtx4090_qname_census",
        lambda model, *, identity_cache_path: copy.deepcopy(live_census),
    )
    monkeypatch.setattr(
        sample_probe, "stable_source_census_projection",
        lambda census: copy.deepcopy(census["stable_projection"]),
    )
    bundle = {
        "source_model_content_sha256": "a" * 64,
        "source_model_upstream_portable_content_sha256": "d" * 64,
        "_validated_source_census_projection": {
            "identity_sha256": "b" * 64,
        },
    }
    assert _validate_live_sample_source_census(
        bundle, model="/model", source_identity_cache="/cache.json",
    ) == live_census
    if changed == "portable":
        live_census["source_census"]["source_model_identity"][
            "content_sha256"
        ] = "c" * 64
    else:
        live_census["stable_projection"]["identity_sha256"] = "c" * 64

    with pytest.raises(
        RTX4090FP8BurnError, match="differs from the validated sample bundle",
    ):
        _validate_live_sample_source_census(
            bundle, model="/model", source_identity_cache="/cache.json",
        )


@pytest.mark.parametrize("changed", ["bundle", "binding", "live"])
def test_sample_bundle_source_bridge_refuses_same_geometry_other_model(changed):
    live = _test_streamed_identity()
    upstream = _test_portable_content(live)
    bundle = {
        "source_model_content_sha256": "a" * 64,
        "source_model_upstream_portable_content_sha256": upstream,
    }
    binding = {"portable_content_sha256": upstream}
    if changed == "bundle":
        bundle["source_model_upstream_portable_content_sha256"] = "f" * 64
    elif changed == "binding":
        binding["portable_content_sha256"] = "f" * 64
    else:
        live = _test_streamed_identity(semantic_tag="different-model")
    with pytest.raises(RTX4090FP8BurnError, match="live model content differ"):
        _validate_sample_bundle_source_binding(
            bundle, binding, live_source_identity=live,
        )


def test_calibration_contract_requires_exact_blake2b128_identity():
    assert _calibration_contract(
        {"nsamples": 32, "seqlen": 1024, "calib_hash": "a" * 32},
        nsamples=32, seqlen=1024, seed=42,
    )["calib_hash"] == "a" * 32
    with pytest.raises(RTX4090FP8BurnError, match="BLAKE2b-128"):
        _calibration_contract(
            {"nsamples": 32, "seqlen": 1024, "calib_hash": "a" * 64},
            nsamples=32, seqlen=1024, seed=42,
        )


def test_probe_imatrix_contract_binds_exact_derived_values(tmp_path):
    from prismaquant.cb_imatrix import canonical_imatrix_sha256

    probe = tmp_path / "probe.pkl"
    col_weights = tmp_path / "col-weights.pkl"
    stats = {
        "model.layers.0.proj": {
            "act_sq_sum": torch.tensor([8.0, 18.0], dtype=torch.float32),
            "n_tokens_seen": 2,
            "in_features": 2,
        },
    }
    with probe.open("wb") as handle:
        pickle.dump({
            "stats": stats,
            "meta": {"calib_hash": "c" * 32},
        }, handle)
    expected = {
        "model.layers.0.proj": torch.tensor([4.0, 9.0], dtype=torch.float32),
    }
    with col_weights.open("wb") as handle:
        pickle.dump(expected, handle)
    contract = _probe_imatrix_contract(probe, col_weights)
    assert contract == {
        "schema": IMATRIX_CONTRACT_SCHEMA,
        "derivation_schema": (
            "prismaquant.cb_imatrix.probe_act_sq_sum_over_tokens.v1"
        ),
        "calibration_hash": "c" * 32,
        "qname_count": 1,
        "qname_census_sha256": canonical_json_sha256(
            ["model.layers.0.proj"], where="probe imatrix qname census",
        ),
        "value_sha256": canonical_imatrix_sha256(expected),
    }

    altered = {"model.layers.0.proj": expected["model.layers.0.proj"].clone()}
    altered["model.layers.0.proj"][0] += 0.25
    with col_weights.open("wb") as handle:
        pickle.dump(altered, handle)
    with pytest.raises(RTX4090FP8BurnError, match="values differ"):
        _probe_imatrix_contract(probe, col_weights)


def _captured_imatrix_probe():
    return {
        "stats": {
            "model.layers.0.proj": {
                "act_sq_sum": torch.tensor(
                    [8.0, 18.0], dtype=torch.float32,
                ),
                "n_tokens_seen": 2,
                "in_features": 2,
            },
        },
        "meta": {"calib_hash": "c" * 32},
    }


def test_derive_col_weights_cli_is_bundle_scoped():
    args = _build_parser().parse_args([
        "derive-col-weights",
        "--sample-merge-bundle", "/run/merged",
        "--output", "/run/cb_col_weights.pkl",
    ])
    assert args.handler is derive_col_weights
    assert args.sample_merge_bundle == "/run/merged"
    assert args.output == "/run/cb_col_weights.pkl"


def test_allocate_cli_requires_complete_sample_bundle():
    args = _build_parser().parse_args([
        "allocate",
        "--plan", "/run/plan.json",
        "--producer-snapshot", "/pq/.prismaquant-runtime-snapshot.json",
        "--model", "/model",
        "--probe", "/run/merged/probe.pkl",
        "--sample-merge-commit", "/run/merged/commit.json",
        "--activation-cache-dir", "/run/merged/activation_cache",
        "--col-weights", "/run/cb_col_weights.pkl",
        "--merged", "/run/aura-merged.pkl",
        "--cost-output", "/run/allocator-cost.pkl",
        "--output-dir", "/run/allocation",
    ])
    assert args.handler is allocate
    assert args.sample_merge_commit == "/run/merged/commit.json"
    assert args.activation_cache_dir == "/run/merged/activation_cache"


def test_derive_col_weights_requires_public_capture_and_exact_calibration(
    monkeypatch, tmp_path,
):
    import prismaquant.sample_parallel_probe as sample_probe

    observed = {}

    def _validate(bundle, *, capture_consumables):
        observed["bundle"] = bundle
        observed["capture_consumables"] = capture_consumables
        return {"_validated_probe_payload": _captured_imatrix_probe()}

    monkeypatch.setattr(
        sample_probe, "validate_sample_parallel_merge_bundle", _validate,
    )
    output = tmp_path / "col-weights.pkl"
    result = derive_col_weights(SimpleNamespace(
        sample_merge_bundle=str(tmp_path / "merged"), output=str(output),
    ))
    assert result == output
    assert observed == {
        "bundle": tmp_path / "merged",
        "capture_consumables": True,
    }
    with output.open("rb") as handle:
        payload = pickle.load(handle)
    assert set(payload) == {"model.layers.0.proj"}
    assert payload["model.layers.0.proj"].dtype == torch.float32
    torch.testing.assert_close(
        payload["model.layers.0.proj"], torch.tensor([4.0, 9.0]),
    )

    invalid = _captured_imatrix_probe()
    invalid["meta"]["calib_hash"] = "not-exact"
    monkeypatch.setattr(
        sample_probe,
        "validate_sample_parallel_merge_bundle",
        lambda *_args, **_kwargs: {"_validated_probe_payload": invalid},
    )
    with pytest.raises(RTX4090FP8BurnError, match="BLAKE2b-128"):
        derive_col_weights(SimpleNamespace(
            sample_merge_bundle=str(tmp_path / "merged"),
            output=str(tmp_path / "invalid.pkl"),
        ))
    assert not (tmp_path / "invalid.pkl").exists()


def test_derive_col_weights_output_race_preserves_competing_file(
    monkeypatch, tmp_path,
):
    import prismaquant.sample_parallel_probe as sample_probe

    monkeypatch.setattr(
        sample_probe,
        "validate_sample_parallel_merge_bundle",
        lambda *_args, **_kwargs: {
            "_validated_probe_payload": _captured_imatrix_probe(),
        },
    )
    output = tmp_path / "col-weights.pkl"
    real_link = sample_probe.os.link

    def _race_link(source, destination):
        assert Path(destination) == output
        output.write_bytes(b"competing-writer")
        return real_link(source, destination)

    monkeypatch.setattr(sample_probe.os, "link", _race_link)
    with pytest.raises(RTX4090FP8BurnError, match="refusing to overwrite"):
        derive_col_weights(SimpleNamespace(
            sample_merge_bundle=str(tmp_path / "merged"), output=str(output),
        ))
    assert output.read_bytes() == b"competing-writer"


def test_allocator_cost_resume_reuses_only_exact_regular_bytes(tmp_path):
    import prismaquant.rtx4090_fp8_burn as burn

    output = tmp_path / "allocator-cost.pkl"
    payload = {"costs": {"model.layers.0.proj": {"BF16": 1.0}}}
    burn._publish_or_reuse_allocator_cost(output, payload, resume=False)
    exact = output.read_bytes()

    burn._publish_or_reuse_allocator_cost(output, payload, resume=True)
    assert output.read_bytes() == exact
    with pytest.raises(RTX4090FP8BurnError, match="output exists"):
        burn._publish_or_reuse_allocator_cost(output, payload, resume=False)
    with pytest.raises(RTX4090FP8BurnError, match="differs from the exact"):
        burn._publish_or_reuse_allocator_cost(
            output, {"costs": {"different": {}}}, resume=True,
        )
    assert output.read_bytes() == exact


@pytest.mark.parametrize("dangling", [False, True])
def test_allocator_cost_resume_refuses_symlink(tmp_path, dangling):
    import prismaquant.rtx4090_fp8_burn as burn

    target = tmp_path / "target.pkl"
    if not dangling:
        target.write_bytes(b"not-the-cost")
    output = tmp_path / "allocator-cost.pkl"
    output.symlink_to(target)
    with pytest.raises(
        RTX4090FP8BurnError, match="not a readable regular file",
    ):
        burn._publish_or_reuse_allocator_cost(
            output, {"costs": {}}, resume=True,
        )
    assert output.is_symlink()


def test_allocate_uses_exact_sealed_bundle_probe_through_child(
    monkeypatch, tmp_path,
):
    import prismaquant.rtx4090_fp8_burn as burn

    validated_probe = {"stats": {}, "meta": {}}
    validated_probe_bytes = pickle.dumps(
        validated_probe, protocol=pickle.HIGHEST_PROTOCOL,
    )
    probe_sha256 = hashlib.sha256(validated_probe_bytes).hexdigest()
    activation_identity = "a" * 64
    portable_identity = "b" * 64
    plan = {
        "plan_sha256": "c" * 64,
        "imatrix": {"exact": True},
        "bindings": {
            "probe": {
                "sha256": probe_sha256,
                "bytes": len(validated_probe_bytes),
            },
            "activation_cache_manifest": {
                "identity_sha256": activation_identity,
            },
            "source_model_identity": {
                "portable_content_sha256": portable_identity,
            },
        },
    }
    live_census = {"linear_entries": {"authoritative": True}}
    sample_bundle = {
        "commit_identity_sha256": "d" * 64,
        "probe_sha256": probe_sha256,
        "probe_bytes": len(validated_probe_bytes),
        "activation_manifest_identity_sha256": activation_identity,
        "_validated_probe_payload": validated_probe,
        "_validated_probe_bytes": validated_probe_bytes,
        "_validated_source_census_projection": live_census,
    }
    observed = {}
    monkeypatch.setattr(burn, "load_campaign_plan", lambda _path: plan)
    monkeypatch.setattr(
        burn, "_validate_burn_runtime_snapshot", lambda _path: {},
    )
    monkeypatch.setattr(burn, "_verify_binding", lambda *_a, **_k: None)
    monkeypatch.setattr(
        burn, "_validate_sample_merge_bundle",
        lambda **kwargs: observed.setdefault("bundle_args", kwargs)
        and sample_bundle,
    )
    monkeypatch.setattr(
        burn, "_validate_sample_bundle_source_binding",
        lambda bundle, binding: observed.setdefault(
            "source_join", (bundle, binding),
        ),
    )

    def _revalidate(**kwargs):
        observed["census_probe"] = kwargs["validated_probe_payload"]
        observed["authoritative_census"] = kwargs["authoritative_census"]

    monkeypatch.setattr(burn, "_revalidate_live_campaign_census", _revalidate)
    monkeypatch.setattr(
        burn,
        "_probe_imatrix_contract",
        lambda *_a, **kwargs: (
            observed.setdefault(
                "imatrix_probe", kwargs["validated_probe_payload"],
            )
            and plan["imatrix"]
        ),
    )
    monkeypatch.setattr(
        burn, "_load_pickle_mapping", lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        burn, "_validate_merged_campaign_provenance", lambda *_a: None,
    )
    monkeypatch.setattr(
        burn, "_allocator_cost", lambda *_a: {"costs": {}},
    )

    def _run_allocator_once(**kwargs):
        command = list(kwargs["command"])
        sealed_path = Path(command[command.index("--probe") + 1])
        observed["sealed_path"] = str(sealed_path)
        observed["pass_fds"] = tuple(kwargs["pass_fds"])
        assert sealed_path.read_bytes() == validated_probe_bytes
        replacement = tmp_path / "replacement-probe.pkl"
        replacement.write_bytes(b"different probe")
        with pytest.raises(OSError):
            os.replace(replacement, sealed_path)
        with pytest.raises(OSError):
            os.write(ALLOCATOR_PROBE_FD, b"mutate")
        assert sealed_path.read_bytes() == validated_probe_bytes
        return Path(kwargs["output_dir"])

    monkeypatch.setattr(burn, "run_allocator_once", _run_allocator_once)
    result = allocate(SimpleNamespace(
        plan=str(tmp_path / "plan.json"),
        producer_snapshot=str(tmp_path / "snapshot.json"),
        model=str(tmp_path / "model"),
        probe=str(tmp_path / "merged" / "probe.pkl"),
        sample_merge_commit=str(tmp_path / "merged" / "commit.json"),
        activation_cache_dir=str(tmp_path / "merged" / "activation_cache"),
        col_weights=str(tmp_path / "cb_col_weights.pkl"),
        merged=str(tmp_path / "aura-merged.pkl"),
        cost_output=str(tmp_path / "allocator-cost.pkl"),
        output_dir=str(tmp_path / "allocation"),
        threads=1,
        resume=False,
    ))
    assert result == tmp_path / "allocation" / "layer_config.json"
    assert observed["sealed_path"] == f"/proc/self/fd/{ALLOCATOR_PROBE_FD}"
    assert observed["pass_fds"] == (ALLOCATOR_PROBE_FD,)
    assert observed["census_probe"] is validated_probe
    assert observed["authoritative_census"] is live_census
    assert observed["imatrix_probe"] is validated_probe
    with pytest.raises(OSError):
        os.fstat(ALLOCATOR_PROBE_FD)


def _stats(*, reverse: bool = False):
    rows = []
    for layer in range(64):
        rows.extend((
            (
                f"model.layers.{layer}.self_attn.q_proj",
                {"n_params": 32, "in_features": 4, "out_features": 8},
            ),
            (
                f"model.layers.{layer}.mlp.down_proj",
                {"n_params": 32, "in_features": 8, "out_features": 4},
            ),
        ))
    if reverse:
        rows.reverse()
    return dict(rows)


def _plan(*, reverse: bool = False):
    portable_content = _test_portable_content(_test_streamed_identity())
    source_body = {
        "schema": SOURCE_IDENTITY_BINDING_SCHEMA,
        "portable_content_sha256": portable_content,
    }
    source_identity_sha256 = canonical_json_sha256(
        source_body, where="test source identity binding",
    )
    source_binding = {
        **source_body,
        "identity_sha256": source_identity_sha256,
        "sha256": source_identity_sha256,
        "bytes": len(json.dumps(
            source_body, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")),
    }
    return build_campaign_plan(
        _stats(reverse=reverse), profile=_Profile(),
        fixed_bf16={
            "lm_head": {
                "reason": "profile_pinned", "source_dtype": "bf16",
                "n_params": 100,
            },
            "mtp.proj": {
                "reason": "mtp_fixed", "source_dtype": "bf16",
                "n_params": 20,
            },
        },
        calibration={
            "calib_hash": "c" * 32, "nsamples": 32,
            "seqlen": 1024, "seed": 42,
        },
        bindings={
            "probe": {"sha256": "1" * 64, "bytes": 101},
            "col_weights": {"sha256": "2" * 64, "bytes": 102},
            "source_model_identity": source_binding,
            "producer_snapshot": {"sha256": "5" * 64, "bytes": 105},
            "common_execution_attestation": {
                "sha256": "6" * 64, "bytes": 106,
                "schema": EXECUTION_ATTESTATION_SCHEMA,
                "identity_sha256": "6" * 64,
                "container_image_digest": "sha256:" + "6" * 64,
            },
            "dataset": {"sha256": "7" * 64, "bytes": 107},
            "sample_merge_commit": {
                "sha256": "9" * 64,
                "bytes": 109,
                "schema": (
                    "prismaquant.sample_parallel_probe."
                    "merge_bundle_commit.v1"
                ),
                "identity_sha256": "a" * 64,
                "cover_identity_sha256": "b" * 64,
                "execution_identity_sha256": "d" * 64,
            },
            "activation_cache_manifest": {
                "sha256": "e" * 64,
                "bytes": 110,
                "schema": (
                    "prismaquant.probe."
                    "sample_activation_cache_merge.v1"
                ),
                "identity_sha256": "f" * 64,
                "cover_identity_sha256": "b" * 64,
                "activation_qname_manifest_sha256": "0" * 64,
                "source_census_sha256": "a" * 64,
            },
        },
        source_dtype_census_sha256="8" * 64,
        imatrix_contract={
            "schema": IMATRIX_CONTRACT_SCHEMA,
            "derivation_schema": (
                "prismaquant.cb_imatrix.probe_act_sq_sum_over_tokens.v1"
            ),
            "calibration_hash": "c" * 32,
            "qname_count": len(_stats()),
            "qname_census_sha256": "b" * 64,
            "value_sha256": "c" * 64,
        },
    )


def test_plan_has_exact_full_menu_and_terminal_contract():
    plan = _plan()
    assert plan["schema"] == PLAN_SCHEMA
    assert tuple(plan["policy"]["formats"]) == FULL_FORMATS
    assert tuple(plan["policy"]["measured_formats"]) == MEASURED_FORMATS
    assert tuple(plan["policy"]["codebook_formats"]) == CB_FORMATS
    assert FULL_FORMATS[-2:] == (NATIVE_FP8_FORMAT, BF16_FORMAT)
    assert BF16_FORMAT not in MEASURED_FORMATS
    assert MEASURED_FORMATS == (
        "FP8_CB_K40",
        "FP8_CB_K44",
        "FP8_CB_K48",
        NATIVE_FP8_FORMAT,
    )
    assert plan["producer"]["streamed_model_cache"] == {
        "max_cache_slots": 2,
        "effective_prefetch_lookahead": 1,
        "require_prefetched_residency": True,
    }
    assert STREAMING_CACHE_MAX_SLOTS == 2
    assert STREAMING_PREFETCH_LOOKAHEAD == 1
    assert STREAMING_REQUIRE_PREFETCHED_RESIDENCY is True

    for qname in plan["body"]["qnames"]:
        assert tuple(plan["maps"]["formats_by_qname"][qname]) == RENDER_FORMATS
        purposes = plan["maps"]["purposes_by_qname"][qname]
        assert set(purposes) == set(MEASURED_FORMATS)
        assert purposes == {
            "FP8_CB_K40": ["panel"],
            "FP8_CB_K44": ["anchor", "panel"],
            "FP8_CB_K48": ["panel"],
            NATIVE_FP8_FORMAT: ["anchor"],
        }
        assert plan["maps"]["unmeasured_formats_by_qname"][qname] == [
            BF16_FORMAT
        ]
        assert tuple(
            plan["maps"]["legal_cb_formats_by_qname"][qname]
        ) == CB_FORMATS


def test_campaign_compile_settings_require_shared_fail_closed_switch(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_COMPILE", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_ATOM_COMPILE", "1")
    monkeypatch.delenv(CB_COMPILE_FAIL_CLOSED_ENV, raising=False)
    with pytest.raises(
        RTX4090FP8BurnError,
        match="PRISMAQUANT_CB_COMPILE_FAIL_CLOSED",
    ):
        _require_compile_settings()
    monkeypatch.setenv(CB_COMPILE_FAIL_CLOSED_ENV, "1")
    assert _require_compile_settings() == _arm_identity(_plan())[
        "compile_settings"
    ]


def test_measure_constructs_exact_resident_two_slot_one_lookahead_runner():
    tree = ast.parse(inspect.getsource(measure))

    def dict_value(mapping, key):
        assert isinstance(mapping, ast.Dict)
        for raw_key, value in zip(mapping.keys, mapping.values, strict=True):
            if isinstance(raw_key, ast.Constant) and raw_key.value == key:
                return value
        raise AssertionError(f"missing AST mapping key {key!r}")

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_streamed_causal_lm"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    max_slots = keywords["max_cache_slots"]
    lookahead = keywords["prefetch_lookahead"]
    require_resident = keywords["require_prefetched_residency"]
    assert isinstance(max_slots, ast.Name)
    assert max_slots.id == "STREAMING_CACHE_MAX_SLOTS"
    assert isinstance(lookahead, ast.Name)
    assert lookahead.id == "STREAMING_PREFETCH_LOOKAHEAD"
    assert isinstance(require_resident, ast.Name)
    assert require_resident.id == "STREAMING_REQUIRE_PREFETCHED_RESIDENCY"

    aura_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_streamed_cb_anchor_aura"
    ]
    assert len(aura_calls) == 1
    aura_keywords = {
        keyword.arg: keyword.value for keyword in aura_calls[0].keywords
    }
    checkpoint_extra = aura_keywords["checkpoint_identity_extra"]
    if isinstance(checkpoint_extra, ast.Name):
        assignments = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == checkpoint_extra.id
                for target in node.targets
            )
        ]
        assert len(assignments) == 1
        checkpoint_extra = assignments[0]
    cache_contract = dict_value(
        checkpoint_extra, "streaming_source_cache",
    )
    checkpoint_require_resident = dict_value(
        cache_contract, "require_prefetched_residency",
    )
    assert isinstance(checkpoint_require_resident, ast.Name)
    assert checkpoint_require_resident.id == (
        "STREAMING_REQUIRE_PREFETCHED_RESIDENCY"
    )


def test_plan_contains_no_out_of_lane_family_string():
    encoded = json.dumps(_plan(), sort_keys=True)
    disallowed = "NV" + "FP4"
    assert disallowed not in encoded


def test_plan_is_deterministic_across_probe_mapping_order():
    forward = _plan()
    reverse = _plan(reverse=True)
    assert forward == reverse
    assert forward["plan_sha256"] == reverse["plan_sha256"]


def test_contiguous_equal_lpt_tie_is_exact_disjoint_cover():
    plan = _plan()
    stripes = plan["stripes"]
    assert [row["layer_range"] for row in stripes] == [[0, 31], [32, 63]]
    assert plan["stripe_balance_proof"] == {
        **plan["stripe_balance_proof"],
        "strategy": "whole_layer_lpt_contiguous_equal_tie",
        "selected_metrics_exactly_equal": True,
        "selected_matches_lpt_loads": True,
    }
    first = set(stripes[0]["qnames"])
    second = set(stripes[1]["qnames"])
    assert first.isdisjoint(second)
    assert first | second == set(plan["body"]["qnames"])
    assert all(".layers." in name for name in first | second)
    assert all(int(name.split(".layers.", 1)[1].split(".", 1)[0]) < 32
               for name in first)
    assert all(int(name.split(".layers.", 1)[1].split(".", 1)[0]) >= 32
               for name in second)
    for metric in ("qnames", "parameters", "estimated_work", "render_cells"):
        assert (
            plan["stripe_balance_proof"]["selected"][0][metric]
            == plan["stripe_balance_proof"]["selected"][1][metric]
        )


def test_plan_validation_rejects_any_menu_edit():
    plan = _plan()
    plan["maps"]["formats_by_qname"][plan["body"]["qnames"][0]].pop()
    without_digest = dict(plan)
    without_digest.pop("plan_sha256")
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    plan["plan_sha256"] = canonical_json_sha256(
        without_digest, where="tampered test plan"
    )
    with pytest.raises(RTX4090FP8BurnError, match="maps"):
        validate_campaign_plan(plan)


def _rehash_plan(plan):
    body = dict(plan)
    body.pop("plan_sha256", None)
    plan["plan_sha256"] = canonical_json_sha256(
        body, where="self-rehashed test campaign plan",
    )
    return plan


def _authoritative_census(source_census):
    entries = {}
    for name, source_dtype in source_census.items():
        if name == "lm_head":
            disposition = LM_HEAD_STATS_ONLY
        elif name.startswith("mtp."):
            disposition = MTP_STATS_ONLY
        else:
            disposition = BODY_STATS_AND_ACTIVATION
        entries[name] = {
            "source_dtype": source_dtype.upper(),
            "disposition": disposition,
        }
    return {
        "source_census": {
            "source_linear_count": len(entries),
            "linear_entries": entries,
        },
    }


def _live_campaign_fixture(tmp_path):
    plan = _plan()
    source_census = {name: "bf16" for name in plan["body"]["qnames"]}
    source_census.update({"lm_head": "bf16", "mtp.proj": "bf16"})
    plan["source_dtype_census_sha256"] = canonical_json_sha256(
        dict(sorted(source_census.items())), where="live source dtype census",
    )
    _rehash_plan(plan)
    validate_campaign_plan(plan)
    probe_stats = _stats()
    probe_stats.update({
        "lm_head": {
            "n_params": 100, "in_features": 10, "out_features": 10,
        },
        "mtp.proj": {
            "n_params": 20, "in_features": 5, "out_features": 4,
        },
    })
    probe = tmp_path / "live-probe.pkl"
    with probe.open("wb") as handle:
        pickle.dump({"stats": probe_stats, "meta": {}}, handle)
    return plan, probe, _authoritative_census(source_census)


_PREPARE_FAILURE_NON_LINEAR_QNAMES = (
    "model.embed_tokens",
    "model.layers.0.input_layernorm",
    "model.layers.0.linear_attn.conv1d",
    "model.layers.0.linear_attn.norm",
    "model.layers.0.post_attention_layernorm",
    "model.layers.1.input_layernorm",
    "model.layers.1.linear_attn.conv1d",
    "model.layers.1.linear_attn.norm",
)


def test_authoritative_linear_census_excludes_prepare_failure_non_linears(
    monkeypatch, tmp_path,
):
    import prismaquant.allocator_candidates as allocator_candidates

    plan, probe, authoritative_census = _live_campaign_fixture(tmp_path)
    generic_scanner_calls = []

    def _poisoned_generic_scanner(_model, _profile):
        generic_scanner_calls.append(True)
        return {
            **{
                name: "bf16"
                for name in plan["body"]["qnames"]
            },
            **{
                name: "bf16"
                for name in _PREPARE_FAILURE_NON_LINEAR_QNAMES
            },
        }

    monkeypatch.setattr(
        allocator_candidates,
        "_scan_source_dtype_manifest",
        _poisoned_generic_scanner,
    )
    source_dtypes, body, fixed = _authoritative_linear_source_census(
        authoritative_census
    )
    assert _authoritative_linear_source_census(
        authoritative_census["source_census"]
    ) == (source_dtypes, body, fixed)
    excluded = set(_PREPARE_FAILURE_NON_LINEAR_QNAMES)
    assert excluded.isdisjoint(source_dtypes)
    assert excluded.isdisjoint(body)
    assert excluded.isdisjoint(fixed)
    _revalidate_live_campaign_census(
        authoritative_census=authoritative_census,
        probe=probe,
        plan=plan,
    )
    assert generic_scanner_calls == []


def test_authoritative_linear_census_refuses_missing_probe_linear(tmp_path):
    plan, _probe, authoritative_census = _live_campaign_fixture(tmp_path)
    missing = str(plan["body"]["qnames"][0])
    probe_stats = _stats()
    probe_stats.pop(missing)
    probe_stats.update({
        "lm_head": {
            "n_params": 100, "in_features": 10, "out_features": 10,
        },
        "mtp.proj": {
            "n_params": 20, "in_features": 5, "out_features": 4,
        },
    })
    with pytest.raises(
        RTX4090FP8BurnError,
        match=rf"live probe/source census mismatch: missing body=\['{missing}'\]",
    ):
        _revalidate_live_campaign_census(
            authoritative_census=authoritative_census,
            probe="unused-probe.pkl",
            plan=plan,
            validated_probe_payload={"stats": probe_stats, "meta": {}},
        )


def test_authoritative_linear_census_refuses_source_dtype_drift(tmp_path):
    plan, probe, authoritative_census = _live_campaign_fixture(tmp_path)
    changed = copy.deepcopy(authoritative_census)
    qname = str(plan["body"]["qnames"][0])
    changed["source_census"]["linear_entries"][qname][
        "source_dtype"
    ] = "F16"
    with pytest.raises(
        RTX4090FP8BurnError,
        match=rf"source Linear {qname} has dtype class 'f16'",
    ):
        _revalidate_live_campaign_census(
            authoritative_census=changed,
            probe=probe,
            plan=plan,
        )


@pytest.mark.parametrize("surface", ["source", "fixed"])
def test_live_census_refuses_self_rehashed_plan_provenance(
    tmp_path, surface,
):
    plan, probe, authoritative_census = _live_campaign_fixture(tmp_path)
    _revalidate_live_campaign_census(
        authoritative_census=authoritative_census,
        probe=probe,
        plan=plan,
    )
    changed = copy.deepcopy(plan)
    if surface == "source":
        changed["source_dtype_census_sha256"] = "d" * 64
    else:
        changed["fixed_bf16_census"][0]["n_params"] += 1
        changed["fixed_bf16_census_sha256"] = canonical_json_sha256(
            changed["fixed_bf16_census"], where="changed fixed census",
        )
    _rehash_plan(changed)
    validate_campaign_plan(changed)
    with pytest.raises(RTX4090FP8BurnError, match="differs from the live"):
        _revalidate_live_campaign_census(
            authoritative_census=authoritative_census,
            probe=probe,
            plan=changed,
        )


@pytest.mark.parametrize(
    ("surface", "message"),
    (
        ("policy", "policy constants"),
        ("producer", "producer constants"),
        ("calibration", "calibration constants"),
        ("bindings", "binding shape"),
        ("fixed", "fixed-BF16"),
        ("body", "body counters"),
        ("stripe", "qname filename"),
    ),
)
def test_self_rehashed_plan_cannot_edit_closed_campaign_surfaces(
    surface, message,
):
    plan = _plan()
    if surface == "policy":
        plan["policy"]["target_bytes"] += 1
    elif surface == "producer":
        plan["producer"]["render_levers"]["weighted_vq"] = False
    elif surface == "calibration":
        plan["calibration"]["nsamples"] = 31
    elif surface == "bindings":
        plan["bindings"]["probe"]["unreviewed"] = True
    elif surface == "fixed":
        plan["fixed_bf16_census"][0]["source_dtype"] = "float16"
        plan["fixed_bf16_census_sha256"] = canonical_json_sha256(
            plan["fixed_bf16_census"], where="tampered fixed census",
        )
    elif surface == "body":
        plan["body"]["parameters"] += 1
    else:
        plan["stripes"][0]["qname_file"] = "stripe-alternate.qnames.txt"
    _rehash_plan(plan)
    with pytest.raises(RTX4090FP8BurnError, match=message):
        validate_campaign_plan(plan)


def _fully_restored_compile_proof(plan, stripe, source_model):
    compile_settings = _arm_identity(plan)["compile_settings"]
    prior = os.environ.get(CB_COMPILE_FAIL_CLOSED_ENV)
    os.environ[CB_COMPILE_FAIL_CLOSED_ENV] = "1"
    try:
        token = begin_cb_compile_execution_proof()
        compiler_proof = finish_cb_compile_execution_proof(token)
    finally:
        if prior is None:
            os.environ.pop(CB_COMPILE_FAIL_CLOSED_ENV, None)
        else:
            os.environ[CB_COMPILE_FAIL_CLOSED_ENV] = prior
    qnames = tuple(str(name) for name in stripe["qnames"])
    checkpoint_binding = {
        "schema": AURA_CHECKPOINT_BINDING_SCHEMA,
        "manifest_schema": AURA_CHECKPOINT_MANIFEST_SCHEMA,
        "identity_schema": AURA_CHECKPOINT_IDENTITY_SCHEMA,
        "identity_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "unit_count": len(qnames),
        "unit_qnames_sha256": canonical_json_sha256(
            sorted(qnames), where="test checkpoint qnames",
        ),
        "compile_settings_sha256": canonical_json_sha256(
            compile_settings, where="test compile settings",
        ),
        "arm_identity_sha256": canonical_json_sha256(
            _arm_identity(plan), where="test arm identity",
        ),
        "streamed_model_identity_sha256": canonical_json_sha256(
            source_model, where="test streamed model identity",
        ),
    }
    return build_campaign_cb_compile_proof(
        compiler_proof,
        compile_settings=compile_settings,
        expected_qnames=qnames,
        rendered_cells=0,
        restored_cells=len(qnames) * len(MEASURED_FORMATS),
        formats_per_unit=len(MEASURED_FORMATS),
        cb_formats_per_unit=len(MEASURED_CB_FORMATS),
        checkpoint_binding=checkpoint_binding,
    )


def _receipt_test_shard(plan, stripe_index):
    stripe = plan["stripes"][stripe_index]
    source_model = _test_streamed_identity(
        root=f"/worker-{stripe_index}/model",
        staged_from=f"/tmp/prismaquant_stage_worker_{stripe_index}",
    )
    assert _test_portable_content(source_model) == plan["bindings"][
        "source_model_identity"
    ]["portable_content_sha256"]
    payload = {
        "n_probes": AURA_N_PROBES,
        "token_scope": AURA_TOKEN_SCOPE,
        "costs": {name: {} for name in stripe["qnames"]},
        "stats": {
            name: {
                "out_features": plan["body"]["shapes"][name][0],
                "in_features": plan["body"]["shapes"][name][1],
                "n_params": math.prod(plan["body"]["shapes"][name]),
            }
            for name in stripe["qnames"]
        },
        "provenance": {
            "production_anchor_renderer": {
                "arm_identity": _arm_identity(plan),
                "source_model": source_model,
            },
            "production_anchor_expected_renders": (
                len(stripe["qnames"]) * len(MEASURED_FORMATS)
            ),
            "production_anchor_rendered_this_invocation": 0,
            "production_anchor_restored_renders": (
                len(stripe["qnames"]) * len(MEASURED_FORMATS)
            ),
        },
    }
    payload["provenance"][CB_COMPILE_PROOF_KEY] = (
        _fully_restored_compile_proof(plan, stripe, source_model)
    )
    _attach_campaign_shard_receipt(
        payload, plan, stripe_index=stripe_index,
        compile_settings=_arm_identity(plan)["compile_settings"],
        model_identity=source_model,
    )
    return payload


def test_campaign_shard_receipts_are_order_independent_exact_stripe_cover():
    plan = _plan()
    shards = [_receipt_test_shard(plan, index) for index in range(2)]
    _validate_campaign_shards(shards, plan)
    _validate_campaign_shards(list(reversed(shards)), plan)
    assert all(
        shard["provenance"][SHARD_RECEIPT_KEY]["schema"]
        == SHARD_RECEIPT_SCHEMA
        for shard in shards
    )
    for shard in shards:
        proof = shard["provenance"][SHARD_RECEIPT_KEY][
            "cb_compile_execution_proof"
        ]
        assert proof["coverage"]["status"] == "restored_strict_checkpoint"
        assert proof["compiler_proof"]["totals"]["attempted_calls"] == 0
        assert proof["atom_route"] == {
            "status": "not_applicable",
            "reason": "campaign_cb_serialization_ldlq_false",
            "ldlq": False,
            "ldlq_scope": "none",
            "compiled_calls": 0,
        }


def test_campaign_shard_receipt_refuses_resealed_atom_execution_claim():
    plan = _plan()
    shard = _receipt_test_shard(plan, 0)
    proof = shard["provenance"][CB_COMPILE_PROOF_KEY]
    proof["atom_route"]["compiled_calls"] = 1
    body = dict(proof)
    body.pop("proof_sha256")
    proof["proof_sha256"] = canonical_json_sha256(
        body, where="tampered campaign compile proof",
    )
    with pytest.raises(
        RTX4090FP8BurnError,
        match="CB compile execution proof is invalid",
    ):
        _validate_campaign_shards(
            [shard, _receipt_test_shard(plan, 1)], plan,
        )


def test_campaign_shard_receipts_refuse_cross_plan_relabeling():
    source_plan = _plan()
    shards = [_receipt_test_shard(source_plan, index) for index in range(2)]
    caller_plan = copy.deepcopy(source_plan)
    caller_plan["bindings"]["dataset"]["sha256"] = "9" * 64
    _rehash_plan(caller_plan)
    validate_campaign_plan(caller_plan)
    with pytest.raises(RTX4090FP8BurnError, match="production arm differs"):
        _validate_campaign_shards(shards, caller_plan)


def test_campaign_shard_receipts_refuse_duplicate_or_cross_stripe_ownership():
    plan = _plan()
    stripe_zero = _receipt_test_shard(plan, 0)
    with pytest.raises(RTX4090FP8BurnError, match="exact receipt-bound cover"):
        _validate_campaign_shards(
            [stripe_zero, copy.deepcopy(stripe_zero)], plan,
        )

    crossed = copy.deepcopy(stripe_zero)
    receipt = crossed["provenance"][SHARD_RECEIPT_KEY]
    receipt["stripe_index"] = 1
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_json_sha256(
        receipt_body, where="tampered cross-stripe receipt",
    )
    with pytest.raises(RTX4090FP8BurnError, match="cost scope is not exact"):
        _validate_campaign_shards(
            [crossed, _receipt_test_shard(plan, 1)], plan,
        )


def test_campaign_shard_receipts_refuse_resealed_different_image_stripe():
    plan = _plan()
    shards = [_receipt_test_shard(plan, index) for index in range(2)]
    receipt = shards[1]["provenance"][SHARD_RECEIPT_KEY]
    receipt["container_image_digest"] = "sha256:" + "7" * 64
    body = dict(receipt)
    body.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_json_sha256(
        body, where="resealed different-image stripe receipt",
    )
    with pytest.raises(RTX4090FP8BurnError, match="receipt differs"):
        _validate_campaign_shards(shards, plan)


@pytest.mark.parametrize("surface", ["costs", "stats"])
def test_campaign_shard_receipt_rejects_payload_mutation_after_issue(surface):
    plan = _plan()
    shards = [_receipt_test_shard(plan, index) for index in range(2)]
    qname = plan["stripes"][0]["qnames"][0]
    if surface == "costs":
        shards[0]["costs"][qname]["post_receipt_tamper"] = 1.0
    else:
        shards[0]["stats"][qname]["n_params"] += 1
    with pytest.raises(RTX4090FP8BurnError, match="receipt differs"):
        _validate_campaign_shards(shards, plan)


def _merged_receipt_fixture(plan):
    shards = [_receipt_test_shard(plan, index) for index in range(2)]
    costs = {
        name: row
        for shard in shards for name, row in shard["costs"].items()
    }
    stats = {
        name: row
        for shard in shards for name, row in shard["stats"].items()
    }
    input_receipts = sorted(
        (
            copy.deepcopy(shard["provenance"][SHARD_RECEIPT_KEY])
            for shard in shards
        ),
        key=lambda row: row["stripe_index"],
    )
    body = {
        "schema": "prismaquant.rtx4090_fp8_burn.merged_aura.v1",
        "global_plan_sha256": plan["plan_sha256"],
        "fixed_bf16_census_sha256": plan["fixed_bf16_census_sha256"],
        "producer_snapshot_sha256": plan["bindings"]["producer_snapshot"][
            "sha256"
        ],
        "common_execution_attestation_sha256": plan["bindings"][
            "common_execution_attestation"
        ]["sha256"],
        "container_image_digest": plan["bindings"][
            "common_execution_attestation"
        ]["container_image_digest"],
        "direct_measured_formats": list(MEASURED_FORMATS),
        "unmeasured_terminal": BF16_FORMAT,
        "input_shard_receipt_schema": SHARD_RECEIPT_SCHEMA,
        "input_shard_receipts": input_receipts,
        "merged_costs_sha256": canonical_json_sha256(
            costs, where="merged RTX4090 costs",
        ),
        "merged_stats_sha256": canonical_json_sha256(
            stats, where="merged RTX4090 stats",
        ),
    }
    receipt = {
        **body,
        "receipt_sha256": canonical_json_sha256(
            body, where="merged RTX4090 provenance receipt",
        ),
    }
    return {
        "n_probes": AURA_N_PROBES,
        "token_scope": AURA_TOKEN_SCOPE,
        "costs": costs,
        "stats": stats,
        "provenance": {"rtx4090_fp8_burn": receipt},
    }


@pytest.mark.parametrize("surface", ["costs", "stats"])
def test_merged_receipts_reject_payload_mutation_after_issue(surface):
    plan = _plan()
    merged = _merged_receipt_fixture(plan)
    _validate_merged_campaign_provenance(merged, plan)
    qname = plan["body"]["qnames"][0]
    if surface == "costs":
        merged["costs"][qname]["post_receipt_tamper"] = 1.0
    else:
        merged["stats"][qname]["n_params"] += 1
    with pytest.raises(RTX4090FP8BurnError, match="differs from payload/plan"):
        _validate_merged_campaign_provenance(merged, plan)


def _rtx_render_identity(qnames, col_weights, *, formats):
    identity = build_production_cache_cb_render_identity(
        {name: list(formats) for name in qnames},
        cb_serialization_context=_cb_context(),
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    return bind_cb_render_identity_source_weights(
        identity,
        {
            name: torch.arange(512, dtype=torch.float32).reshape(2, 256)
            + index
            for index, name in enumerate(qnames)
        },
    )


def _rtx_merged_payload():
    qnames = ("model.layers.0.proj", "model.layers.1.proj")
    col_weights = {
        name: torch.arange(256, dtype=torch.float32) + index
        for index, name in enumerate(qnames)
    }
    sparse = _rtx_render_identity(
        qnames, col_weights, formats=MEASURED_CB_FORMATS
    )
    expanded = _rtx_render_identity(
        qnames, col_weights, formats=CB_FORMATS
    )
    source_records = {
        name: {
            "shape": sparse["source_weights_shapes"][name],
            "sha256": sparse["source_weights_content_sha256"][name],
        }
        for name in qnames
    }
    arm_identity = {"campaign": "rtx4090-test"}
    renderer = {
        "arm_identity": arm_identity,
        "formats_by_qname": {
            name: list(MEASURED_FORMATS) for name in qnames
        },
        "cb_render_identity": sparse,
        "source_weights": {
            "complete": True,
            "scope": "sparse_anchor_plan",
            "records": source_records,
            "identity_sha256": canonical_json_sha256(
                source_records, where="RTX test source records",
            ),
        },
    }
    purposes = {
        name: {
            "FP8_CB_K40": ["panel"],
            "FP8_CB_K44": ["anchor", "panel"],
            "FP8_CB_K48": ["panel"],
            NATIVE_FP8_FORMAT: ["anchor"],
        }
        for name in qnames
    }
    costs = {}
    for unit_index, name in enumerate(qnames, start=1):
        rows = {}
        for format_name in MEASURED_CB_FORMATS:
            rung = int(format_name.rsplit("K", 1)[1])
            value = unit_index * math.exp(-0.02 * rung)
            rows[format_name] = {
                "predicted_dloss": value,
                "predicted_dloss_stderr": value / 100.0,
                "x2_per_probe": [value * 2.0],
                "dw_source": "production_render",
                "output_mse_measured": False,
                "cost_source": "aura",
                "production_anchor_measured": True,
                "production_anchor_zero": False,
                "weight_mse_diagnostic": value * 7.0,
                "weight_mse_diagnostic_normalization": "mean_per_weight",
                "weight_mse_is_cost_input": False,
            }
        rows[NATIVE_FP8_FORMAT] = {
            "predicted_dloss": unit_index * 0.01,
            "predicted_dloss_stderr": unit_index * 0.0001,
            "x2_per_probe": [unit_index * 0.02],
            "dw_source": "production_render",
            "output_mse_measured": False,
            "cost_source": "aura",
            "production_anchor_measured": True,
            "production_anchor_zero": False,
        }
        costs[name] = rows
    return {
        "schema": "prismaquant.aura_cost.v1",
        "n_probes": AURA_N_PROBES,
        "formats": list(RENDER_FORMATS),
        "token_scope": AURA_TOKEN_SCOPE,
        "stats": {
            name: {
                "n_params": 512,
                "in_features": 256,
                "out_features": 2,
            }
            for name in qnames
        },
        "costs": costs,
        "provenance": {
            "production_anchor_renderer": renderer,
            "production_anchor_render_purposes": purposes,
            "production_anchor_unmeasured_formats_by_qname": {
                name: [BF16_FORMAT] for name in qnames
            },
            "production_anchor_sparse_render_identity": sparse,
            "cb_render_identity": expanded,
        },
    }, qnames


def _allocator_plan(merged, qnames):
    return {
        "body": {
            "qnames": list(qnames),
            "shapes": {
                name: [
                    merged["stats"][name]["out_features"],
                    merged["stats"][name]["in_features"],
                ]
                for name in qnames
            },
        },
        "plan_sha256": "p" * 64,
    }


def test_allocator_cost_preserves_every_direct_campaign_rung():
    merged, qnames = _rtx_merged_payload()
    before = {
        name: {
            format_name: pickle.dumps(
                merged["costs"][name][format_name],
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            for format_name in MEASURED_FORMATS
        }
        for name in qnames
    }
    result = _allocator_cost(
        merged,
        _allocator_plan(merged, qnames),
    )

    assert tuple(result["formats"]) == FULL_FORMATS
    imputed = set(CB_FORMATS) - set(MEASURED_FORMATS)
    assert imputed == set()
    for name in qnames:
        rows = result["costs"][name]
        assert set(rows) == set(FULL_FORMATS)
        for format_name in MEASURED_FORMATS:
            assert pickle.dumps(
                rows[format_name], protocol=pickle.HIGHEST_PROTOCOL,
            ) == before[name][format_name]
            assert pickle.dumps(
                merged["costs"][name][format_name],
                protocol=pickle.HIGHEST_PROTOCOL,
            ) == before[name][format_name]
        for format_name in imputed:
            assert rows[format_name]["production_anchor_measured"] is False
            assert rows[format_name][
                "extrapolated_not_rendered_measurement"
            ] is True
            assert rows[format_name]["cost_source"] == (
                "anchored_aura_extrapolation"
            )
        assert rows[BF16_FORMAT]["cost_source"] == "source_passthrough"
        assert rows[BF16_FORMAT]["predicted_dloss"] == 0.0

    stamp = result["provenance"]["rtx4090_fp8_burn_allocator_cost"]
    assert tuple(stamp["direct_measured_formats"]) == MEASURED_FORMATS
    assert set(stamp["imputed_formats"]) == imputed
    hull = result["provenance"]["lower_convex_hull"]
    assert hull["cost_surface"] == (
        "fitted_imputation_law_before_direct_measurement_overlay"
    )
    assert hull["not_a_hull_over_final_overlaid_costs"] is True
    meta = result["meta"]
    assert meta["unit_count"] == len(qnames)
    assert meta["cell_count"] == len(qnames) * len(FULL_FORMATS)
    assert meta["cell_semantics_counts"] == {
        "direct_measured": len(qnames) * len(MEASURED_FORMATS),
        "anchored_cb_imputed": len(qnames) * len(imputed),
        "source_passthrough_terminal": len(qnames),
    }
    assert "direct final table" in meta["cost_semantics"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("n_probes", None, "probe count is absent"),
        ("n_probes", 31, "probe count is absent"),
        ("token_scope", None, "token scope is absent"),
        ("token_scope", "completion_only", "token scope is absent"),
    ),
)
def test_allocator_cost_fail_closes_merged_probe_and_token_contract(
    field, value, message,
):
    merged, qnames = _rtx_merged_payload()
    if value is None:
        merged.pop(field)
    else:
        merged[field] = value
    with pytest.raises(RTX4090FP8BurnError, match=message):
        _allocator_cost(
            merged,
            _allocator_plan(merged, qnames),
        )


@pytest.mark.parametrize("format_name", MEASURED_FORMATS)
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("predicted_dloss", float("nan")),
        ("predicted_dloss", -0.01),
        ("predicted_dloss_stderr", float("inf")),
        ("predicted_dloss_stderr", -0.001),
    ),
)
def test_allocator_cost_refuses_corrupt_direct_measurement(
    format_name, field, value,
):
    merged, qnames = _rtx_merged_payload()
    merged["costs"][qnames[0]][format_name][field] = value
    with pytest.raises(
        RTX4090FP8BurnError,
        match=rf"direct {format_name} {field} must be finite and nonnegative",
    ):
        _allocator_cost(
            merged,
            _allocator_plan(merged, qnames),
        )
