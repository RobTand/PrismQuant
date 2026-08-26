from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch
from tools.serve_fingerprint import fingerprint, performance_stack_fingerprint

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.cost_streaming import portable_streamed_model_content_identity
from prismaquant.prismasnap import PRISMASNAP_ALGORITHM, PrismaSnapSearchConfig
from prismaquant.prismasnap_contract import require_verified_prismasnap_if_present
from prismaquant.prismasnap_validation import (
    PROVENANCE_JSON,
    attest_fold_fidelity,
    validate_prismasnap_provenance_payload,
)
import prismaquant.prismasnap_validation as validation


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path, *, kl: float = 1e-4
) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    source_shard = source / "model-00001-of-00001.safetensors"
    save_file(
        {"model.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
        str(source_shard),
    )
    source_weight_map = {"model.weight": source_shard.name}
    _write_json(
        source / "model.safetensors.index.json", {"weight_map": source_weight_map}
    )
    _write_json(
        source / "config.json",
        {
            "model_type": "bert",
            "hidden_size": 2,
            "intermediate_size": 4,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "vocab_size": 64,
        },
    )
    from transformers import AutoConfig

    source_config = AutoConfig.from_pretrained(
        source, local_files_only=True
    ).to_dict()
    source_identity: dict[str, object] = {
        "schema": "prismaquant.streamed_model.identity.v1",
        "source": str(source.resolve()),
        "resolved_commit": None,
        "config": source_config,
        "weight_map": source_weight_map,
        "checkpoint_weight_map": source_weight_map,
        "shards": [
            {
                "path": str(source_shard.resolve()),
                "size": source_shard.stat().st_size,
                "sha256": _sha256(source_shard),
            }
        ],
    }
    source_identity["content_sha256"] = canonical_json_sha256(
        {
            "config": source_identity["config"],
            "weight_map": source_identity["weight_map"],
            "shards": source_identity["shards"],
            "checkpoint_weight_map": source_identity["checkpoint_weight_map"],
        },
        where="test source identity",
    )
    source_identity_path = tmp_path / "source_identity.json"
    _write_json(source_identity_path, source_identity)
    portable = portable_streamed_model_content_identity(
        source_identity, where="test portable source identity"
    )

    checkpoint = tmp_path / "snapped"
    checkpoint.mkdir()
    shard = checkpoint / "model-00001-of-00001.safetensors"
    save_file({"model.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, str(shard))
    _write_json(
        checkpoint / "model.safetensors.index.json",
        {"weight_map": {"model.weight": shard.name}},
    )
    checkpoint_identity = validation._checkpoint_content_identity(checkpoint)
    producer_files = {"prismaquant/prismasnap.py": "1" * 64}
    provenance: dict[str, object] = {
        "schema": "prismaquant.prismasnap.provenance.v1",
        "state": "MATERIALIZED",
        "algorithm": PRISMASNAP_ALGORITHM,
        "purely_additive_source_preparation": True,
        "serve_time_changes": False,
        "source_portable_content_sha256": portable["portable_content_sha256"],
        "source_local_content_sha256": source_identity["content_sha256"],
        "source_model": str(source.resolve()),
        "probe_sha256": "2" * 64,
        "calibration": {
            "calib_hash": "fixture-calibration",
            "dataset": "fixture-text",
            "nsamples": 8,
            "seqlen": 512,
            "calibration_modality": "text",
        },
        "plan_sha256": "3" * 64,
        "scales_sha256": "4" * 64,
        "producer": {
            "git_commit": "5" * 40,
            "source_sha256": canonical_json_sha256(
                producer_files, where="test producer files"
            ),
            "source_files": producer_files,
            "container_rootfs_sha256": "6" * 64,
            "container_attested": True,
        },
        "search": PrismaSnapSearchConfig().as_dict(),
        "coverage": {
            "body_layers": [0],
            "excluded_prefixes": ["model.visual.", "mtp."],
            "seams": 3,
            "transformed_tensors": 1,
            "materialized_changed_tensors": 1,
        },
        "fp64_invariance": {
            "fp64_invariance_max_abs": 0.0,
            "threshold": 1e-10,
            "domain": "pre_cast_fp64_algebra",
            "required_bf16_fold_kl_max": 5e-4,
        },
        "seam_summary": [
            {
                "layer": 0,
                "kind": kind,
                "graph_sha256": "7" * 64,
                "groups": 1,
                "groups_moved": 1,
                "improvement_fraction": 0.1,
            }
            for kind in ("input_norm", "post_attention_norm", "up_down")
        ],
        "output": dict(checkpoint_identity),
    }
    provenance["provenance_sha256"] = canonical_json_sha256(
        provenance, where="test provenance"
    )
    _write_json(checkpoint / PROVENANCE_JSON, provenance)

    def serve_manifest(model: Path, argv: list[str]) -> dict[str, object]:
        gold_files = {
            "tools/measure_vllm_full_kl.py": {"bytes": 1, "sha256": "8" * 64}
        }
        manifest: dict[str, object] = {
            "schema": "prismaquant.serve_manifest/1",
            "model": str(model),
            "launch_argv": argv,
            "quantization": None,
            "speculative_config": None,
            "measurement_tool": "measure_vllm_full_kl",
            "producer_identity": {
                "schema": "prismaquant.gold_producer_identity/1",
                "measurement_tool": "measure_vllm_full_kl",
                "git_commit": "a" * 40,
                "git_tree": "9" * 40,
                "git_dirty": False,
                "source_files": gold_files,
                "source_files_sha256": canonical_json_sha256(
                    gold_files, where="test gold files"
                ),
            },
        }
        manifest["performance_stack_fingerprint"] = performance_stack_fingerprint(
            manifest
        )
        manifest["serve_fingerprint"] = fingerprint(manifest)
        return manifest

    teacher_payload = tmp_path / "teacher.pt"
    starts = list(range(8))
    topk_ids = torch.arange(4, dtype=torch.int64).view(1, 1, 4).expand(8, 511, 4)
    topk_lps = torch.tensor(
        [-1.0, -2.0, -3.0, -4.0], dtype=torch.float32
    ).view(1, 1, 4).expand(8, 511, 4)
    torch.save(
        {
            "score_positions": "all",
            "prompt_top_k": 4,
            "topk_ids": topk_ids,
            "topk_lps": topk_lps,
            "calib_ids": torch.zeros((8, 512), dtype=torch.int64),
            "starts": starts,
            "model": str(source),
            "n_samples": 8,
            "seqlen": 512,
            "vocab_size": 64,
        },
        teacher_payload,
    )
    teacher_meta = tmp_path / "teacher.json"
    teacher_argv = [
        "python",
        "tools/measure_vllm_full_kl.py",
        "--mode",
        "teacher",
        "--model",
        str(source.resolve()),
        "--output",
        str(teacher_payload.resolve()),
        "--meta-output",
        str(teacher_meta.resolve()),
        "--dtype",
        "bfloat16",
        "--score-positions",
        "all",
        "--n-samples",
        "8",
        "--seqlen",
        "512",
        "--prompt-top-k",
        "4",
    ]
    teacher_serve = serve_manifest(source, teacher_argv)
    _write_json(
        teacher_meta,
        {
            "mode": "teacher",
            "model": str(source),
            "output": str(teacher_payload),
            "score_positions": "all",
            "prompt_top_k": 4,
            "n_samples": 8,
            "seqlen": 512,
            "vocab_size": 64,
            "starts": starts,
            "teacher_shape": [8, 511, 4],
            "corpus": {"corpus_sha256": "c" * 64, "total_tokens": 8192},
            "serve_manifest": teacher_serve,
            "serve_fingerprint": teacher_serve["serve_fingerprint"],
            "spec_decode_detected": False,
        },
    )
    student = tmp_path / "student.json"
    student_argv = [
        "python",
        "tools/measure_vllm_full_kl.py",
        "--mode",
        "student",
        "--model",
        str(checkpoint.resolve()),
        "--output",
        str(student.resolve()),
        "--teacher-payload",
        str(teacher_payload.resolve()),
        "--dtype",
        "bfloat16",
        "--score-positions",
        "all",
        "--n-samples",
        "8",
        "--seqlen",
        "512",
        "--prompt-top-k",
        "4",
    ]
    student_serve = serve_manifest(checkpoint, student_argv)
    _write_json(
        student,
        {
            "mode": "student",
            "model": str(checkpoint),
            "teacher_model": str(source),
            "teacher_payload": str(teacher_payload),
            "teacher_payload_sha256": _sha256(teacher_payload),
            "quantization": None,
            "score_positions": "all",
            "prompt_top_k": 4,
            "n_samples": 8,
            "seqlen": 512,
            "vocab_size": 64,
            "n_positions": 8 * 511,
            "kl_mean": kl,
            "kl_p99": kl,
            "kl_max": kl,
            "kl_per_sample": [kl] * 8,
            "serve_fingerprint": student_serve["serve_fingerprint"],
            "serve_manifest": student_serve,
            "spec_decode_detected": False,
        },
    )
    return checkpoint, student, teacher_meta, source_identity_path


def test_fold_attestation_is_content_bound_idempotent_and_admitted(tmp_path: Path) -> None:
    checkpoint, student, teacher, source_identity = _fixture(tmp_path)
    result = attest_fold_fidelity(checkpoint, student, teacher, source_identity)
    assert result["state"] == "VERIFIED"
    assert result["fold_fidelity"]["passed"] is True
    validate_prismasnap_provenance_payload(
        result, require_verified=True, where="test"
    )
    require_verified_prismasnap_if_present(checkpoint)
    assert attest_fold_fidelity(checkpoint, student, teacher, source_identity) == result


def test_fold_attestation_rejects_threshold_failure_and_pipeline_fails_closed(
    tmp_path: Path,
) -> None:
    checkpoint, student, teacher, source_identity = _fixture(tmp_path, kl=5.0001e-4)
    with pytest.raises(RuntimeError, match="exceeds"):
        attest_fold_fidelity(checkpoint, student, teacher, source_identity)
    with pytest.raises(RuntimeError, match="not in an admitted state"):
        require_verified_prismasnap_if_present(checkpoint)


def test_fold_attestation_rejects_changed_checkpoint_bytes(tmp_path: Path) -> None:
    checkpoint, student, teacher, source_identity = _fixture(tmp_path)
    shard = checkpoint / "model-00001-of-00001.safetensors"
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="bytes changed"):
        attest_fold_fidelity(checkpoint, student, teacher, source_identity)


def test_provenance_rejects_extra_fields_and_boolean_numeric_bypass(
    tmp_path: Path,
) -> None:
    checkpoint, _student, _teacher, _source_identity = _fixture(tmp_path)
    payload = json.loads((checkpoint / PROVENANCE_JSON).read_text(encoding="utf-8"))
    payload["unreviewed_override"] = True
    payload["provenance_sha256"] = validation._provenance_digest(payload)
    with pytest.raises(RuntimeError, match="fields differ"):
        validate_prismasnap_provenance_payload(
            payload, require_verified=False, where="test"
        )

    payload.pop("unreviewed_override")
    payload["fp64_invariance"]["fp64_invariance_max_abs"] = False
    payload["provenance_sha256"] = validation._provenance_digest(payload)
    with pytest.raises(RuntimeError, match="invariance"):
        validate_prismasnap_provenance_payload(
            payload, require_verified=False, where="test"
        )


def test_fold_attestation_rejects_changed_original_source_bytes(tmp_path: Path) -> None:
    checkpoint, student, teacher, source_identity = _fixture(tmp_path)
    identity = json.loads(source_identity.read_text(encoding="utf-8"))
    source_shard = Path(identity["shards"][0]["path"])
    source_shard.write_bytes(source_shard.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="size changed|content changed"):
        attest_fold_fidelity(checkpoint, student, teacher, source_identity)


def _null_floor_receipt(
    tmp_path: Path,
    student: Path,
    checkpoint: Path,
    *,
    kls: tuple[float, float] = (6.0e-4, 6.5e-4),
    teacher_payload_sha256: str | None = None,
) -> Path:
    student_payload = json.loads(student.read_text(encoding="utf-8"))
    provenance = json.loads(
        (checkpoint / "prismasnap_provenance.json").read_text(encoding="utf-8")
    )
    receipt = {
        "schema": "prismaquant.prismasnap.null_floor_receipt.v1",
        "model_source": provenance["source_model"],
        "source_portable_content_sha256": provenance[
            "source_portable_content_sha256"
        ],
        "teacher_payload_sha256": (
            teacher_payload_sha256
            if teacher_payload_sha256 is not None
            else student_payload["teacher_payload_sha256"]
        ),
        "measurement_contract": {
            key: student_payload[key]
            for key in ("score_positions", "prompt_top_k", "n_samples", "seqlen")
        },
        "arms": [
            {
                "arm": name,
                "magnitude": magnitude,
                "perturbed_2d_tensors": 496,
                "kl_mean": value,
                "student_result_sha256": "3" * 64,
                "serve_fingerprint": "4" * 64,
                "splice_receipt_sha256": "5" * 64,
            }
            for name, magnitude, value in (
                ("half_ulp", 2.0**-8, kls[0]),
                ("full_ulp", 2.0**-7, kls[1]),
            )
        ],
    }
    path = tmp_path / "null_floor_receipt.json"
    _write_json(path, receipt)
    return path


def test_null_floor_receipt_derives_fold_threshold(tmp_path: Path) -> None:
    checkpoint, student, teacher, source_identity = _fixture(tmp_path, kl=6.2e-4)
    with pytest.raises(RuntimeError, match="exceeds"):
        attest_fold_fidelity(checkpoint, student, teacher, source_identity)
    receipt = _null_floor_receipt(tmp_path, student, checkpoint)
    result = attest_fold_fidelity(
        checkpoint,
        student,
        teacher,
        source_identity,
        null_floor_receipt_path=receipt,
    )
    assert result["state"] == "VERIFIED"
    fold = result["fold_fidelity"]
    assert fold["threshold"] == max(5e-4, 2.0 * 6.5e-4)
    derivation = fold["threshold_derivation"]
    assert derivation["null_floor_kl_mean"] == 6.5e-4
    assert derivation["plan_threshold"] == 5e-4
    validate_prismasnap_provenance_payload(
        result, require_verified=True, where="test"
    )
    require_verified_prismasnap_if_present(checkpoint)
    assert (
        attest_fold_fidelity(
            checkpoint,
            student,
            teacher,
            source_identity,
            null_floor_receipt_path=receipt,
        )
        == result
    )


def test_null_floor_receipt_requires_saturation_agreement(tmp_path: Path) -> None:
    checkpoint, student, teacher, source_identity = _fixture(tmp_path, kl=6.2e-4)
    receipt = _null_floor_receipt(
        tmp_path, student, checkpoint, kls=(1.0e-4, 6.5e-4)
    )
    with pytest.raises(RuntimeError, match="not saturated"):
        attest_fold_fidelity(
            checkpoint,
            student,
            teacher,
            source_identity,
            null_floor_receipt_path=receipt,
        )


def test_null_floor_receipt_binds_teacher_payload(tmp_path: Path) -> None:
    checkpoint, student, teacher, source_identity = _fixture(tmp_path, kl=6.2e-4)
    receipt = _null_floor_receipt(
        tmp_path, student, checkpoint, teacher_payload_sha256="6" * 64
    )
    with pytest.raises(RuntimeError, match="different teacher payload"):
        attest_fold_fidelity(
            checkpoint,
            student,
            teacher,
            source_identity,
            null_floor_receipt_path=receipt,
        )


def test_null_floor_derivation_cannot_be_edited_after_verification(
    tmp_path: Path,
) -> None:
    checkpoint, student, teacher, source_identity = _fixture(tmp_path, kl=6.2e-4)
    receipt = _null_floor_receipt(tmp_path, student, checkpoint)
    result = attest_fold_fidelity(
        checkpoint,
        student,
        teacher,
        source_identity,
        null_floor_receipt_path=receipt,
    )
    payload = dict(result)
    fold = dict(payload["fold_fidelity"])
    derivation = dict(fold["threshold_derivation"])
    derivation["null_floor_kl_mean"] = 5.0e-3
    fold["threshold_derivation"] = derivation
    payload["fold_fidelity"] = fold
    payload["provenance_sha256"] = validation._provenance_digest(payload)
    with pytest.raises(RuntimeError, match="floor does not equal its arm maximum"):
        validate_prismasnap_provenance_payload(
            payload, require_verified=True, where="test"
        )
