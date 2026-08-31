from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant.trellis_formats import E2M1_FAMILY
from prismaquant.trellis_producer import encode_trellis_one_linear


M = importlib.import_module(
    "research.qtip_native_nvfp4_2026-08-30.arm_e_quality_campaign"
)
NATIVE = importlib.import_module(
    "research.qtip_native_nvfp4_2026-08-30.native_nvfp4_ldlq"
)


def _manifest(tmp_path: Path, *, mode: str = M.QWEN_MODE):
    value = {
        "schema": M.MANIFEST_SCHEMA,
        "campaign_id": "arm-e-test",
        "mode": mode,
        "research_opt_in": M.ARM_E.RESEARCH_OPT_IN,
        "execution": {
            "device": "cpu",
            "host": "test-host",
            "container_identity": "sha256:" + "1" * 64,
            "prismaquant_checkout": str(M._REPO_ROOT),
            "prismaquant_commit": "2" * 40,
        },
        "output": {
            "root": str(tmp_path / "out"),
            "durable_root_uri": "sparky:/tmp/arm-e-test",
        },
        "recipe": {
            **M._FIXED_RECIPE,
            "terminal_metric_mode": "diag_block_D",
            "max_input_block_size": 4096,
            "sb_chunk": 1,
            "backend": "eager",
            "buffer_blocks": 1,
            "glm_algebra_witness_rows": 2,
        },
        "seeds": [
            {"label": "primary", "input_seed": 0x1234, "output_seed": 0x5678}
        ],
    }
    if mode == M.QWEN_MODE:
        value["input"] = {
            "kind": M.QWEN_MODE,
            "model_id": "Qwen/Qwen3-0.6B",
            "weight_path": "/input/weight.safetensors",
            "weight_key": "model.layers.0.self_attn.q_proj.weight",
            "activations_path": "/input/activations.pt",
            "activations_key": "inputs",
            "calibration_manifest": "/input/calibration.json",
        }
    else:
        value["input"] = {
            "kind": M.GLM_MODE,
            "corpus_manifest": "/input/glm/manifest.json",
            "selected_tensors": [],
            "limit": None,
        }
    return value


def test_manifest_is_versioned_closed_and_mode_specific(tmp_path):
    manifest = _manifest(tmp_path)
    assert M.validate_manifest(manifest) == manifest

    unknown = copy.deepcopy(manifest)
    unknown["future_default"] = True
    with pytest.raises(ValueError, match="unknown=.*future_default"):
        M.validate_manifest(unknown)

    wrong_kind = copy.deepcopy(manifest)
    wrong_kind["input"]["kind"] = M.GLM_MODE
    with pytest.raises(ValueError, match="input.kind"):
        M.validate_manifest(wrong_kind)

    bool_alias = copy.deepcopy(manifest)
    bool_alias["recipe"]["tailbite_candidates"] = True
    with pytest.raises(ValueError, match="tailbite_candidates"):
        M.validate_manifest(bool_alias)


@pytest.mark.parametrize(
    ("shape", "expected_q256", "expected_bpw"),
    [
        ((2048, 1024), 992, 4.377437591552734),
        ((4096, 2048), 1008, 4.438612937927246),
        ((2048, 4096), 1016, 4.470870018005371),
        ((12288, 4096), 1016, 4.4691033363342285),
        ((4096, 12288), 1016, 4.4697747230529785),
    ],
)
def test_exact_native_byte_frontier_includes_complete_wire_padding(
    shape, expected_q256, expected_bpw
):
    native_bytes = M.native_payload_bytes(shape)
    plan = M.exact_native_budget_frontier(
        shape, arm_a_bytes=native_bytes, arm_c_bytes=native_bytes
    )
    assert plan["selected_body_rate_q256"] == expected_q256
    assert plan["footprint"]["total_bytes"] <= native_bytes
    assert plan["footprint"]["exact_bpw"] == pytest.approx(expected_bpw)
    assert set(plan["alphabets"]) == {"3"}
    assert set(plan["schedule"]) == {3, 4}


def test_qwen_alignment_forces_q992_not_nominal_q1016():
    shape = (2048, 1024)
    native_bytes = M.native_payload_bytes(shape)
    selected = M.exact_native_budget_frontier(
        shape, arm_a_bytes=native_bytes, arm_c_bytes=native_bytes
    )
    assert selected["selected_body_rate_q256"] == 992
    schedule = M.uniform_column_schedule(1024, 1016, family=E2M1_FAMILY)
    footprint = M.trellis_tensor_payload_breakdown(
        shape,
        family=E2M1_FAMILY,
        body_rate_q256=1016,
        layout=M.LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets=M.canonical_highrate_alphabets(schedule),
    )
    assert footprint["total_bytes"] > native_bytes
    assert footprint["exact_bpw"] == pytest.approx(4.502437591552734)


@pytest.mark.parametrize(
    ("columns", "expected"),
    [(1024, 1024), (2048, 2048), (4096, 4096), (12288, 4096)],
)
def test_transform_block_policy(columns, expected):
    assert M.automatic_input_block_size(columns, 4096) == expected


def test_k12288_dense_factorization_fails_preflight_without_allocating(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("planning allocated a tensor")

    monkeypatch.setattr(torch, "empty", forbidden)
    plan = M.dense_producer_feasibility((4096, 12288))
    assert plan["dense_fp32_hessian_bytes"] == 603_979_776
    assert plan["estimate_is_measurement"] is False
    assert plan["current_dense_producer_executable"] is False
    with pytest.raises(ValueError, match="refusing dense Arm E producer"):
        M.require_dense_producer_feasible((4096, 12288))


def test_quality_execution_refuses_cpu_while_preflight_contract_remains_valid(
    tmp_path, monkeypatch
):
    manifest = M.validate_manifest(_manifest(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="declared CUDA device"):
        M.require_gpu_campaign_execution(manifest)
    cuda_manifest = copy.deepcopy(manifest)
    cuda_manifest["execution"]["device"] = "cuda:0"
    cuda_manifest["recipe"]["backend"] = "triton"
    with pytest.raises(ValueError, match="available CUDA"):
        M.require_gpu_campaign_execution(cuda_manifest)


def test_full_glm_preflight_labels_current_k12288_refusal(tmp_path):
    manifest = M.validate_manifest(_manifest(tmp_path, mode=M.GLM_MODE))
    inputs = M.PreflightInputs(
        M.GLM_MODE,
        (
            M.InputUnit("dense.down", "dense", (4096, 12288), "a" * 64),
            M.InputUnit("routed.down", "routed", (4096, 2048), "b" * 64),
        ),
        {"full_census_selected": True},
    )
    report = M.preflight_report(
        manifest,
        {"path": "manifest.json", "file_sha256": "c" * 64,
         "semantic_identity_sha256": "d" * 64},
        inputs,
        {"identity_sha256": "e" * 64},
    )
    assert report["execution_readiness"] == "refused_current_dense_producer_shape"
    assert report["full_glm_census_requested"] is True
    assert report["full_glm_census_executable"] is False
    assert report["claim_boundary"]["completed_campaign"] is False


def test_native_arm_hessian_entry_preserves_activation_wrapper_bytes():
    generator = torch.Generator().manual_seed(20260830)
    weight = torch.randn(2, 16, generator=generator)
    activations = torch.randn(24, 16, generator=generator)
    _x, hessian, _damp = NATIVE.damped_hessian(
        activations, 16, torch.device("cpu")
    )
    from_activations = NATIVE.qtip_native_arm(weight, activations)
    from_hessian = NATIVE.qtip_native_arm_from_hessian(weight, hessian)
    assert NATIVE.fields_sha256(from_activations.fields) == NATIVE.fields_sha256(
        from_hessian.fields
    )
    assert torch.equal(
        from_activations.reconstruction, from_hessian.reconstruction
    )
    assert from_activations.terminal_blocks == from_hessian.terminal_blocks


def test_metric_availability_separates_qwen_activations_from_glm_importance():
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(3, 16, generator=generator)
    reconstruction = weight + 0.1
    activations = torch.randn(5, 16, generator=generator)
    hessian = activations.T @ activations + torch.eye(16)
    qwen = M.quality_metrics(
        weight,
        reconstruction,
        regularized_hessian=hessian,
        activations=activations,
    )
    assert qwen["activation_output"]["available"] is True
    assert qwen["raw_importance_weighted"]["available"] is False

    importance = torch.linspace(0.0, 1.0, 16)
    diagonal, _contract = M.regularized_glm_diagonal(importance)
    glm = M.quality_metrics(
        weight,
        reconstruction,
        regularized_diagonal=diagonal,
        raw_importance=importance,
    )
    assert glm["activation_output"] == {
        "available": False,
        "reason": M.GLM_ACTIVATION_UNAVAILABLE_REASON,
    }
    assert glm["raw_importance_weighted"]["available"] is True


def test_qwen_load_refuses_same_shape_swap_after_preflight(tmp_path):
    weight_path = tmp_path / "weight.pt"
    replacement_path = tmp_path / "replacement.pt"
    activations_path = tmp_path / "activations.pt"
    calibration_path = tmp_path / "calibration.json"
    original_weight = torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256)
    replacement_weight = original_weight.flip(0).contiguous()
    activations = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256)
    torch.save({"weight": original_weight}, weight_path)
    torch.save({"weight": replacement_weight}, replacement_path)
    torch.save({"inputs": activations}, activations_path)
    calibration_body = {
        "schema": NATIVE.CALIBRATION_SCHEMA,
        "dataset": "unit-test",
        "capture_precision": "float32",
        "calibration_hash": "1" * 64,
        "nsamples": 4,
        "seqlen": 1,
        "seed": 7,
    }
    calibration = {
        **calibration_body,
        "identity_sha256": NATIVE._canonical_sha256(calibration_body),
    }
    calibration_path.write_text(json.dumps(calibration))
    manifest = _manifest(tmp_path)
    manifest["input"].update({
        "weight_path": str(weight_path),
        "weight_key": "weight",
        "activations_path": str(activations_path),
        "activations_key": "inputs",
        "calibration_manifest": str(calibration_path),
    })
    manifest = M.validate_manifest(manifest)
    inputs = M.preflight_inputs(manifest)
    assert inputs.units[0].source_weight_sha256 == inputs.provenance["weight"][
        "tensor_sha256"
    ]

    # Atomic rename preserves shape and pathname while changing the inode and
    # exact bytes after preflight.
    replacement_path.replace(weight_path)
    with pytest.raises(ValueError, match="changed after preflight"):
        M._load_unit_tensors(manifest, inputs, inputs.units[0])


def test_qwen_safetensors_pinned_load_returns_owned_materialized_tensor(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "weight.safetensors"
    original = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    replacement = original.flip(0).contiguous()
    save_file({"weight": original}, path)
    loaded, provenance = M._load_qwen_tensor_bound(path, "weight")
    assert provenance["tensor_sha256"] == NATIVE.tensor_sha256(original)
    save_file({"weight": replacement}, path)
    assert torch.equal(loaded, original)
    assert loaded.untyped_storage().data_ptr() != replacement.untyped_storage().data_ptr()


def test_glm_pinned_load_refuses_whole_artifact_swap_with_same_pair(tmp_path):
    weight = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    importance = torch.arange(4, dtype=torch.float32)
    weight_raw = weight.view(torch.uint8).numpy().tobytes()
    importance_raw = importance.view(torch.uint8).numpy().tobytes()
    artifact = tmp_path / "corpus.safetensors"
    original_bytes = weight_raw + importance_raw + b"old"
    artifact.write_bytes(original_bytes)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("fixture manifest\n")
    entry = SimpleNamespace(
        name="tensor.weight",
        importance_key="tensor.importance",
        source_weight_shape=(2, 4),
        importance_shape=(4,),
        source_weight_sha256=hashlib.sha256(weight_raw).hexdigest(),
        importance_sha256=hashlib.sha256(importance_raw).hexdigest(),
    )
    corpus = SimpleNamespace(
        entries=(entry,),
        artifact_path=artifact,
        manifest_path=manifest_path,
        manifest={
            "file_size_bytes": len(original_bytes),
            "file_sha256": hashlib.sha256(original_bytes).hexdigest(),
        },
        _layout=SimpleNamespace(
            data_start=0,
            tensors={
                entry.name: {"data_offsets": [0, len(weight_raw)]},
                entry.importance_key: {
                    "data_offsets": [
                        len(weight_raw), len(weight_raw) + len(importance_raw)
                    ]
                },
            },
        ),
    )
    unit = M.InputUnit(
        entry.name, "dense", (2, 4), entry.source_weight_sha256
    )
    pinned = M._pin_finalized_glm_corpus(corpus)
    inputs = M.PreflightInputs(
        M.GLM_MODE,
        (unit,),
        {"manifest_file_sha256": M._file_sha256(manifest_path)},
        corpus,
        pinned,
    )
    try:
        loaded_weight, loaded_importance = M._load_unit_tensors(
            {"input": {}}, inputs, unit
        )
        assert torch.equal(loaded_weight, weight)
        assert torch.equal(loaded_importance, importance)

        # Preserve both selected tensors byte-for-byte while swapping only an
        # unselected whole-artifact byte range under the original pathname.
        replacement = tmp_path / "replacement.safetensors"
        replacement.write_bytes(weight_raw + importance_raw + b"new")
        replacement.replace(artifact)
        with pytest.raises(ValueError, match="changed"):
            M._load_unit_tensors({"input": {}}, inputs, unit)
    finally:
        inputs.close()


def _semantic_result_fixture(*, wire_sha256: str, control_sha256: str):
    telemetry = {
        "scope": "offline_quality_campaign_noncomparative",
        "preflight_plan": {"current_dense_producer_executable": True},
        "phase_seconds": {"load_inputs": 1.0},
        "total_measured_phase_seconds": 1.0,
        "torch_cuda_peak_allocated_bytes": 1,
        "cpu_factorization_fallback_observed": False,
        "gpu_utilization_used_as_diagnostic": False,
        "serving_or_throughput_claim": False,
    }
    producer_body = {
        "schema": M.ARM_E.BLOCKLDL_COMBINED_ARTIFACT_SCHEMA,
        "wire_bytes": 1,
        "wire_identity_sha256": wire_sha256,
        "decoded_weight_sha256": "e" * 64,
        "decoded_codes_sha256": "d" * 64,
        "same_byte_reparse_verified": True,
        "block_ldl": {"cross_block_feedback_nonzero_count": 1},
        "producer_eligible": False,
    }
    producer_receipt = {
        **producer_body,
        "identity_sha256": M.ARM_E._canonical_sha256(producer_body),
    }
    body = {
        "schema": M.RESULT_SCHEMA,
        "status": "matched_quality_isolate_complete",
        "campaign_claim_identity_sha256": "a" * 64,
        "source_closure_identity_sha256": "b" * 64,
        "mode": M.QWEN_MODE,
        "tensor": {"tensor_sha256": "c" * 64},
        "hessian_contract": {"construction": "fixture"},
        "metric_availability": {},
        "rate_plan": {"selected_body_rate_q256": 1016},
        "transform_geometry": {},
        "controls": {
            "A_scalar_native_nvfp4": {
                "fields_sha256": control_sha256,
                "metrics": {"weight": {"nsse": 0.0}},
            },
            "C_stock_blockldl_native_nvfp4": {
                "fields_sha256": control_sha256,
                "metrics": {"weight": {"nsse": 0.0}},
            },
        },
        "arm_e_by_seed": [{
            "wire": {
                "sha256": wire_sha256,
                "publication_status": "published_no_replace",
            },
            "metrics": {"weight": {"nsse": 0.0}},
            "producer_receipt": producer_receipt,
        }],
        "feasibility_telemetry": telemetry,
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    return {**body, "identity_sha256": M._identity_sha256(body)}


def test_semantic_replay_rejects_zero_wire_and_fabricated_quality(tmp_path):
    schedule = M.uniform_column_schedule(256, 1016, family=E2M1_FAMILY)
    zero_artifact = encode_trellis_one_linear(
        torch.zeros(1, 256),
        torch.ones(256),
        family=E2M1_FAMILY,
        body_rate_q256=1016,
        schedule=schedule,
        layout=M.LAYOUT_TIGHT_OFFSETS,
        alphabets=M.canonical_highrate_alphabets(schedule),
        scale_rule="static_6",
        sb_chunk=1,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
    )
    forged = _semantic_result_fixture(
        wire_sha256=hashlib.sha256(zero_artifact.wire_bytes).hexdigest(),
        control_sha256="f" * 64,
    )
    reproduced = _semantic_result_fixture(
        wire_sha256="1" * 64,
        control_sha256="2" * 64,
    )
    with pytest.raises(ValueError, match="semantic attestation was not reproduced"):
        M._require_semantic_reproduction(
            forged, reproduced, path=tmp_path / "result.json"
        )


def test_semantic_replay_normalizes_only_publication_status(tmp_path):
    existing = _semantic_result_fixture(
        wire_sha256="1" * 64, control_sha256="2" * 64
    )
    reproduced = copy.deepcopy(existing)
    reproduced["arm_e_by_seed"][0]["wire"][
        "publication_status"
    ] = "resumed_identical_existing"
    reproduced_body = dict(reproduced)
    reproduced_body.pop("identity_sha256")
    reproduced["identity_sha256"] = M._identity_sha256(reproduced_body)
    M._require_semantic_reproduction(
        existing, reproduced, path=tmp_path / "result.json"
    )


def test_published_wire_is_reopened_reserialized_and_decoded(tmp_path):
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(1, 256, generator=generator)
    schedule = M.uniform_column_schedule(256, 1016, family=E2M1_FAMILY)
    alphabets = M.canonical_highrate_alphabets(schedule)
    artifact = encode_trellis_one_linear(
        weight,
        torch.ones(256),
        family=E2M1_FAMILY,
        body_rate_q256=1016,
        schedule=schedule,
        layout=M.LAYOUT_TIGHT_OFFSETS,
        alphabets=alphabets,
        scale_rule="static_6",
        sb_chunk=1,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
    )
    footprint = M.trellis_tensor_payload_breakdown(
        weight.shape,
        family=E2M1_FAMILY,
        body_rate_q256=1016,
        layout=M.LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets=alphabets,
    )
    path = tmp_path / "wire.trellis"
    assert M._publish_or_verify_bytes(path, artifact.wire_bytes) == "published_no_replace"
    assert M._publish_or_verify_bytes(path, artifact.wire_bytes) == "resumed_identical_existing"
    with pytest.raises(ValueError, match="differs under the same claim"):
        M._publish_or_verify_bytes(path, artifact.wire_bytes + b"x")
    verified = M.verify_published_wire(
        path,
        expected_blob=artifact.wire_bytes,
        expected_decoded=artifact.decoded_weight,
        expected_footprint_bytes=footprint["total_bytes"],
    )
    assert verified["same_byte_reopen_verified"] is True
    assert verified["same_byte_reserialize_verified"] is True
    assert verified["same_byte_decode_verified"] is True


def test_source_closure_detects_post_import_drift(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("original\n")
    pin = tmp_path / "prismaquant/gridbook_runtime/gridbook_runtime_pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({
        "schema": "test.pin.v1",
        "required_abi_features": {},
    }))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(M, "_source_paths", lambda: {"source.py": source})
    monkeypatch.setattr(M, "_IMPORTED_SOURCE_SHA256", {"source.py": digest})
    first = M.require_source_closure(tmp_path)
    assert first["source_sha256"] == {"source.py": digest}
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="changed since module import"):
        M.require_source_closure(tmp_path)


def test_persistent_claim_refuses_another_campaign_identity(tmp_path):
    destination = tmp_path / "receipt.json"
    identity = {"schema": M.CLAIM_SCHEMA, "manifest": "a" * 64}
    with M.ATOMIC.exclusive_publication_claim(destination, identity=identity):
        pass
    with pytest.raises(M.ATOMIC.PublicationError, match="identity differs"):
        with M.ATOMIC.exclusive_publication_claim(
            destination,
            identity={"schema": M.CLAIM_SCHEMA, "manifest": "b" * 64},
        ):
            pass


def test_completed_receipt_resume_requires_exact_manifest_member_census(tmp_path):
    manifest_provenance = {"semantic_identity_sha256": "a" * 64}
    input_provenance = {"kind": M.QWEN_MODE, "identity_sha256": "b" * 64}
    source_closure = {"identity_sha256": "c" * 64}
    execution = {
        "declared_host": "test-host",
        "observed_hostname": "test-host",
        "container_identity": "sha256:" + "f" * 64,
        "device": {"type": "cuda", "index": 0},
        "torch_version": "test",
        "cuda_toolkit_version": "test",
        "command": ["arm_e_quality_campaign.py", "--manifest", "test.json"],
    }
    publication = M._campaign_publication_record("sparky:/tmp/arm-e-test")
    body = {
        "schema": M.RECEIPT_SCHEMA,
        "status": "quality_campaign_complete",
        "campaign_claim_identity_sha256": "d" * 64,
        "manifest": manifest_provenance,
        "mode": M.QWEN_MODE,
        "input_provenance": input_provenance,
        "source_closure": source_closure,
        "execution": execution,
        "acceptance_contract": M.ACCEPTANCE_CONTRACT,
        "summary": {},
        # A locally rewritten receipt can be self-digested and internally
        # coherent while omitting every manifest-declared tensor artifact.
        "published_members": [{
            "kind": "unrelated",
            "relative_path": "unrelated.bin",
            "bytes": 1,
            "sha256": "e" * 64,
        }],
        "publication": publication,
        "claim_boundary": {
            "quality_only": True,
            "activation_output_model_quality": False,
            "gridbook_runtime_executed": False,
            "served": False,
            "performance_claim": False,
            "producer_eligible": False,
            "runtime_pin_changed": False,
            "production_contract_changed": False,
        },
    }
    receipt = {**body, "identity_sha256": M._identity_sha256(body)}
    path = tmp_path / "receipt.json"
    path.write_bytes(M._canonical_bytes(receipt, pretty=True))
    expected_members = {
        "tensors/000-one/result.json": "tensor_result_commit_marker",
        "tensors/000-one/primary.trellis": "canonical_trellis_wire",
    }
    with pytest.raises(ValueError, match="member census differs from manifest"):
        M._verify_complete_receipt(
            path,
            claim_identity_sha256="d" * 64,
            source_closure=source_closure,
            manifest_provenance=manifest_provenance,
            input_provenance=input_provenance,
            expected_members=expected_members,
            expected_execution=execution,
            expected_publication=publication,
            mode=M.QWEN_MODE,
            root=tmp_path,
        )

    body["execution"] = {}
    receipt = {**body, "identity_sha256": M._identity_sha256(body)}
    path.write_bytes(M._canonical_bytes(receipt, pretty=True))
    with pytest.raises(ValueError, match="execution identity differs"):
        M._verify_complete_receipt(
            path,
            claim_identity_sha256="d" * 64,
            source_closure=source_closure,
            manifest_provenance=manifest_provenance,
            input_provenance=input_provenance,
            expected_members=expected_members,
            expected_execution=execution,
            expected_publication=publication,
            mode=M.QWEN_MODE,
            root=tmp_path,
        )


def test_resume_refuses_forged_completed_seed_prefix(tmp_path):
    unit = M.InputUnit("one.weight", "qwen_one_linear", (256, 256), None)
    manifest = M.validate_manifest(_manifest(tmp_path))
    manifest["seeds"].append({
        "label": "holdout",
        "input_seed": 0x1235,
        "output_seed": 0x5679,
    })
    native_bytes = M.native_payload_bytes(unit.shape)
    rate_plan = M.exact_native_budget_frontier(
        unit.shape, arm_a_bytes=native_bytes, arm_c_bytes=native_bytes
    )
    metrics = {
        "weight": {"nsse": 0.1, "snr_db": 10.0, "numerator": 1.0, "denominator": 10.0},
        "regularized_hessian_proxy": {
            "available": True,
            "basis": "untransformed_original_linear",
            "representation": "dense",
            "nsse": 0.1,
            "snr_db": 10.0,
            "numerator": 1.0,
            "denominator": 10.0,
        },
        "activation_output": {
            "available": True,
            "reason": M.QWEN_ACTIVATION_AVAILABLE_REASON,
            "nsse": 0.1,
            "snr_db": 10.0,
            "numerator": 1.0,
            "denominator": 10.0,
        },
        "raw_importance_weighted": {
            "available": False,
            "reason": "raw importance is defined only by the GLM corpus contract",
        },
    }
    payload = {
        "bytes": native_bytes,
        "n_weights": 256 * 256,
        "bits_per_weight": 8.0 * native_bytes / (256 * 256),
    }
    control = {
        "carrier": "compressed_tensors_native_nvfp4_fields",
        "fields_sha256": "a" * 64,
        "payload": payload,
        "metrics": metrics,
    }
    body = {
        "schema": M.RESULT_SCHEMA,
        "status": "matched_quality_isolate_complete",
        "campaign_claim_identity_sha256": "b" * 64,
        "source_closure_identity_sha256": "c" * 64,
        "mode": M.QWEN_MODE,
        "tensor": {
            "name": unit.name,
            "population": unit.population,
            "shape": list(unit.shape),
            "corpus_raw_weight_sha256": None,
        },
        "hessian_contract": {},
        "metric_availability": {
            "activation_output": True,
            "raw_importance_weighted": False,
            "regularized_hessian_proxy": True,
        },
        "rate_plan": rate_plan,
        "transform_geometry": {
            "input_block_size": 256,
            "output_block_size": 256,
            "physical_blockldl_terminal_columns": 256,
        },
        "controls": {
            "A_scalar_native_nvfp4": control,
            "C_stock_blockldl_native_nvfp4": control,
        },
        # A forged coherent prefix must not skip the second declared seed.
        "arm_e_by_seed": [{"seed": manifest["seeds"][0]}],
        "feasibility_telemetry": {},
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    value = {**body, "identity_sha256": M._identity_sha256(body)}
    result_path = tmp_path / f"tensors/000-{M._slug(unit.name)}/result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(M._canonical_bytes(value, pretty=True))
    with pytest.raises(ValueError, match="seed census differs"):
        M._resume_unit_result(
            tmp_path,
            index=0,
            unit=unit,
            mode=M.QWEN_MODE,
            claim_identity_sha256="b" * 64,
            source_closure_identity_sha256="c" * 64,
            expected_seeds=manifest["seeds"],
            recipe=manifest["recipe"],
            reproduced_result=value,
        )
