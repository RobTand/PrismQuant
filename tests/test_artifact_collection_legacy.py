from __future__ import annotations

import json

import pytest

from prismaquant.artifact_collection import ArtifactCollectionError, load_record
from prismaquant.artifact_collection_legacy import (
    audit_legacy_export,
    verify_legacy_audit,
)
from prismaquant.cost_stage_checkpoint import canonical_json_sha256


def _write_fixture(tmp_path, *, conflicting_sidecar: bool = False):
    quant = {
        "provenance": {
            "tensor_formats": {
                "body.a": "NVFP4_CB_K1",
                "body.b": "NVFP4_CB_K3",
                "lm_head": "FP8_E4M3",
            },
            "cb_targets": 2,
            "serialized_payload": {
                "index_bytes": 30,
                "fp4_scale_bytes": 10,
                "fp8_row_scale_bytes": 0,
                "global_scale_bytes": 0,
                "input_global_scale_bytes": 0,
                "codebook_sidecar_bytes": 9 if conflicting_sidecar else 8,
                "n_tensors": 2,
            },
            "artifact_inventory": {
                "scope": "all_regular_files_recursive",
                "file_bytes": {"model.bin": 100, "config.json": 10},
                "export_directory_bytes": 110,
                "cb_tensor_payload_bytes": 40,
                "cb_codebook_sidecar_bytes": 8,
                "cb_serialized_payload_bytes": 48,
            },
        }
    }
    shapes = {
        "body.a.weight": {"shape": [2, 2], "dtype": "BF16"},
        "mtp.fc.weight": {"shape": [2, 3], "dtype": "BF16"},
        "mtp.norm.weight": {"shape": [2], "dtype": "BF16"},
    }
    quant_path = tmp_path / "quant_config.json"
    shapes_path = tmp_path / "shapes.json"
    quant_path.write_text(json.dumps(quant), encoding="utf-8")
    shapes_path.write_text(json.dumps(shapes), encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.bin").write_bytes(b"x" * 100)
    (artifact / "config.json").write_bytes(b"x" * 10)
    (artifact / "README.md").write_bytes(b"x" * 5)
    return quant_path, shapes_path, artifact


def test_legacy_audit_keeps_assignment_namespace_and_byte_scopes_separate(tmp_path):
    quant, shapes, artifact = _write_fixture(tmp_path)
    record = audit_legacy_export(
        quant_config_path=quant,
        shapes_path=shapes,
        artifact_root=artifact,
    )
    verify_legacy_audit(record)

    assignment = record["payload"]["assignment_census"]
    assert assignment["format_assignment_unit_count"] == 3
    assert assignment["cb_assignment_unit_count"] == 2
    assert assignment["auxiliary_assignment_unit_count"] == 1
    assert assignment["namespace_matrix_module_count"] == 1
    assert assignment["format_or_namespace_matrix_module_count"] == 4

    physical = record["payload"]["physical_tensor_census"]
    assert physical["physical_model_tensor_count"] == 3
    assert physical["namespace_tensor_count"] == 2
    assert physical["namespace_tensor_bytes"] == 16
    assert physical["namespace_matrix_weight_bytes"] == 12
    assert physical["namespace_support_tensor_bytes"] == 4

    ledger = record["payload"]["byte_ledger"]
    assert ledger["body_serialized_bytes"] == 48
    assert ledger["fixed_residual_recorded_bytes"] == 46
    assert ledger["recorded"]["total_bytes"] == 110
    assert ledger["observed"]["total_bytes"] == 115
    assert ledger["recursive_drift_bytes"] == 5
    assert ledger["fixed_residual_observed_bytes"] == 51
    assert ledger["observed"]["extra_files"] == ["README.md"]
    assert str(artifact.resolve()) not in {
        locator
        for locations in record["locators"].values()
        for locator in locations
    }


def test_legacy_audit_without_artifact_root_does_not_invent_observation(tmp_path):
    quant, shapes, _ = _write_fixture(tmp_path)
    record = audit_legacy_export(quant_config_path=quant, shapes_path=shapes)
    ledger = record["payload"]["byte_ledger"]
    assert ledger["observed"] is None
    assert ledger["recursive_drift_bytes"] is None
    assert ledger["fixed_residual_observed_bytes"] is None


def test_legacy_audit_rejects_conflicting_duplicate_payload_ledgers(tmp_path):
    quant, shapes, artifact = _write_fixture(tmp_path, conflicting_sidecar=True)
    with pytest.raises(ArtifactCollectionError, match="sidecar bytes differ"):
        audit_legacy_export(
            quant_config_path=quant,
            shapes_path=shapes,
            artifact_root=artifact,
        )


def test_generic_loader_runs_strict_legacy_payload_validation(tmp_path):
    quant, shapes, _ = _write_fixture(tmp_path)
    record = audit_legacy_export(quant_config_path=quant, shapes_path=shapes)
    record["payload"]["assignment_census"]["ambiguous_unit_count"] = 4
    record["payload_sha256"] = canonical_json_sha256(
        record["payload"], where="tampered legacy fixture"
    )
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ArtifactCollectionError, match="field set differs"):
        load_record(path)


def test_legacy_file_ledger_rejects_normalization_collisions(tmp_path):
    quant_path, shapes, artifact = _write_fixture(tmp_path)
    quant = json.loads(quant_path.read_text(encoding="utf-8"))
    quant["provenance"]["artifact_inventory"]["file_bytes"] = {
        "a/b": 50,
        "a//b": 60,
    }
    quant_path.write_text(json.dumps(quant), encoding="utf-8")
    with pytest.raises(ArtifactCollectionError, match="multiple names normalize"):
        audit_legacy_export(
            quant_config_path=quant_path,
            shapes_path=shapes,
            artifact_root=artifact,
        )
