from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant.cb_anchored_cost import write_exportable_artifacts
from prismaquant.nvfp4_cb_footprint import assignment_serialization_sha256
import prismaquant.dsv4_w8a16_export_handoff as handoff


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_allocator(
    root: Path,
    assignment: dict[str, str],
    *,
    assignment_sha256: str,
) -> None:
    root.mkdir()
    layer = {
        **assignment,
        "__prismaquant__": {
            "cb_serialized_payload": {
                "schema": "fixture.cb.serialization.v1",
                "codebook_content_sha256": {"fixture": "a" * 64},
                "codebook_source_by_format": {"FP8_CB_K28": "learned"},
            },
        },
    }
    selection = {
        "feasible": True,
        **handoff.DSV4_W8A16_APPROVED_SELECTION,
        "whole_artifact_budget": {
            "selection_assignment_sha256": assignment_sha256,
            "selection_tensor_payload_bytes": handoff.DSV4_W8A16_APPROVED_SELECTION[
                "selection_tensor_payload_bytes"
            ],
            "selection_whole_artifact_upper_bound_bytes": (
                handoff.DSV4_W8A16_APPROVED_SELECTION[
                    "selection_whole_artifact_upper_bound_bytes"
                ]
            ),
        },
    }
    (root / "layer_config.json").write_text(json.dumps(layer))
    (root / "selection.json").write_text(json.dumps(selection))
    (root / "pareto.knees.json").write_text(json.dumps({
        "primary": "quality",
        "quality": {"achieved_bits": 2.75},
    }))


def _publication(
    tmp_path: Path,
    name: str,
    assignment: dict[str, str],
    *,
    provenance: dict[str, object],
    assignment_sha256: str,
) -> Path:
    allocator = tmp_path / f"{name}-allocator"
    col = tmp_path / f"{name}-col.pkl"
    col.write_bytes(b"exact-imatrix")
    _write_allocator(
        allocator, assignment, assignment_sha256=assignment_sha256
    )
    out = tmp_path / name
    write_exportable_artifacts(
        out,
        allocator_output_dir=allocator,
        cb_col_weights_path=col,
        provenance=provenance,
    )
    return out


def _fixture(monkeypatch, tmp_path: Path):
    assignment = {
        f"model.layers.{index}.self_attn.wq_a": "FP8_BLOCK_UE8M0_SOURCE"
        for index in range(120)
    }
    assignment_sha256 = assignment_serialization_sha256(assignment)
    monkeypatch.setattr(handoff, "DSV4_TOTAL_UNITS", 120)
    monkeypatch.setattr(
        handoff, "DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256", assignment_sha256
    )

    raw = _publication(
        tmp_path,
        "raw",
        assignment,
        provenance={"budget_bytes": handoff.DSV4_W8A16_APPROVED_BUDGET_BYTES},
        assignment_sha256=assignment_sha256,
    )
    monkeypatch.setattr(
        handoff,
        "DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256",
        _sha256(raw / "layer_config.json"),
    )
    monkeypatch.setattr(
        handoff,
        "DSV4_W8A16_APPROVED_SELECTION_SHA256",
        _sha256(raw / "selection.json"),
    )
    monkeypatch.setattr(
        handoff,
        "DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256",
        _sha256(raw / "cb_col_weights.pkl"),
    )

    runtime = {
        "schema": "prismaquant.gridbook-runtime-pin.v3",
        "repository": "https://example.invalid/gridbook.git",
        "commit": "a" * 40,
        "version": "0.8.5",
        "version_is_release": True,
        "runtime_contract_schema": "gridbook.runtime-contract.v3",
        "required_abi_features": {
            "routed_moe_per_role_codebook_lut": 1,
            "source_fp8_block128_w8a16": 1,
        },
        "serving_route": handoff.ROUTE_GRIDBOOK_FP8_SOURCE_W8A16,
    }
    raw_proof = {
        "publication": str(raw.resolve()),
        "layer_config_sha256": handoff.DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256,
        "selection_sha256": handoff.DSV4_W8A16_APPROVED_SELECTION_SHA256,
        "cb_col_weights_sha256": (
            handoff.DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256
        ),
        "assignment_sha256": assignment_sha256,
        "selection": dict(handoff.DSV4_W8A16_APPROVED_SELECTION),
    }
    attestation = {
        "approved_assignment_sha256": assignment_sha256,
        "readmitted_assignment_sha256": assignment_sha256,
        "full_qname_format_map_equal": True,
        "selection": dict(handoff.DSV4_W8A16_APPROVED_SELECTION),
    }
    provenance = {
        "budget_bytes": handoff.DSV4_W8A16_APPROVED_BUDGET_BYTES,
        "cpu_replay": {
            "schema": handoff.DSV4_W8A16_READMISSION_SCHEMA,
            "measurement_invoked": False,
            "no_gpu_measurement_or_render": True,
            "approved_raw_publication": raw_proof,
            "gridbook_runtime_pin": {
                key: runtime[key]
                for key in (
                    "schema", "repository", "commit", "version",
                    "version_is_release", "runtime_contract_schema",
                    "required_abi_features",
                )
            },
        },
        "approved_raw_assignment_attestation": attestation,
    }
    publication = _publication(
        tmp_path,
        "readmitted",
        assignment,
        provenance=provenance,
        assignment_sha256=assignment_sha256,
    )

    source = tmp_path / "source"
    source.mkdir()
    source_identity = tmp_path / "source-identity.json"
    source_identity.write_text("{}")
    bundle = tmp_path / "bundle.pqcb"
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(handoff, "_verify_runtime_contract", lambda: runtime)
    monkeypatch.setattr(handoff, "ROUTE_PENDING_PASSTHROUGH_FORMATS", frozenset())
    monkeypatch.setattr(
        "prismaquant.cost_streaming.validate_cached_streamed_model_identity",
        lambda *_args, **_kwargs: {
            "content_sha256": "b" * 64,
            "shards": [{"sha256": "c" * 64}],
        },
    )
    monkeypatch.setattr(
        "prismaquant.dspark_source_metadata.discover_dspark_source_overlay_from_artifact",
        lambda _path: SimpleNamespace(
            construction_units={"mtp.fixture": "FP8_BLOCK_UE8M0_SOURCE"}
        ),
    )
    monkeypatch.setattr(
        handoff,
        "_verify_bundle",
        lambda path, _payload: {
            "path": str(path.resolve()),
            "file_sha256": _sha256(path),
            "bundle_content_sha256": "d" * 64,
            "codebook_count": 1,
        },
    )
    return publication, raw, source, source_identity, bundle


def _verify(tmp_path: Path, fixture):
    publication, raw, source, source_identity, bundle = fixture
    return handoff.verify_dsv4_w8a16_export_handoff(
        publication_dir=publication,
        approved_raw_publication_dir=raw,
        source_model_dir=source,
        source_identity_path=source_identity,
        codebook_bundle_path=bundle,
        output_path=tmp_path / "artifact",
        repo_root=Path(__file__).resolve().parents[1],
    )


def test_exact_w8a16_handoff_returns_read_only_receipt(monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _verify(tmp_path, fixture)
    assert receipt["schema"] == handoff.DSV4_W8A16_EXPORT_HANDOFF_SCHEMA
    assert receipt["assignment_sha256"] == (
        handoff.DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256
    )
    assert receipt["unit_count"] == 120
    assert receipt["fp8_block_w8a16_count"] == 120
    assert receipt["source_checkpoint"]["content_sha256"] == "b" * 64
    closure = receipt["frozen_export_source_closure"]
    assert closure["schema"] == (
        handoff.DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA
    )
    assert closure["files_sha256"] == handoff._FROZEN_EXPORT_SOURCE_SHA256
    assert closure["identity_sha256"] == handoff.canonical_json_sha256(
        {
            "schema": closure["schema"],
            "files_sha256": closure["files_sha256"],
        },
        where="test exporter/source closure",
    )
    assert "frozen_export_code_sha256" not in receipt
    assert receipt["output_absent"] is True
    assert not (tmp_path / "artifact").exists()


def test_handoff_refuses_publication_byte_drift(monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    publication = fixture[0]
    with (publication / "selection.json").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(
        handoff.W8A16ExportHandoffError, match="atomic manifest"
    ):
        _verify(tmp_path, fixture)


def test_handoff_refuses_existing_output(monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    (tmp_path / "artifact").mkdir()
    with pytest.raises(
        handoff.W8A16ExportHandoffError, match="already exists"
    ):
        _verify(tmp_path, fixture)


def test_handoff_refuses_route_pending_after_overlay(monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        handoff,
        "ROUTE_PENDING_PASSTHROUGH_FORMATS",
        frozenset({"FP8_BLOCK_UE8M0_SOURCE"}),
    )
    with pytest.raises(
        handoff.W8A16ExportHandoffError, match="route-pending"
    ):
        _verify(tmp_path, fixture)


def test_tracked_frozen_export_sources_match_reviewed_bytes():
    root = Path(__file__).resolve().parents[1]
    expected_paths = {
        "prismaquant/artifact_completeness.py",
        "prismaquant/cb_export_config.py",
        "prismaquant/cb_source_decode.py",
        "prismaquant/dspark_source_metadata.py",
        "prismaquant/export_nvfp4_cb_streaming.py",
        "prismaquant/export_output_safety.py",
        "prismaquant/layer_streaming.py",
        "prismaquant/model_profiles/__init__.py",
        "prismaquant/model_profiles/base.py",
        "prismaquant/model_profiles/deepseek_v4.py",
        "prismaquant/model_profiles/registry.py",
        "prismaquant/model_profiles/specs/deepseek_v4.json",
        "prismaquant/nvfp4_cb_footprint.py",
        "prismaquant/nvfp4_cb_formats.py",
        "prismaquant/production_weight_cache.py",
    }
    assert set(handoff._FROZEN_EXPORT_SOURCE_SHA256) == expected_paths
    observed = handoff._verify_frozen_export_source_closure(root)
    assert observed["files_sha256"] == handoff._FROZEN_EXPORT_SOURCE_SHA256
    assert observed["schema"] == (
        handoff.DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA
    )


def test_frozen_export_source_closure_refuses_dependency_drift(
    monkeypatch,
    tmp_path,
):
    relative = "prismaquant/dspark_source_metadata.py"
    root = Path(__file__).resolve().parents[1]
    copied = tmp_path / relative
    copied.parent.mkdir(parents=True)
    copied.write_bytes((root / relative).read_bytes() + b"\n")
    monkeypatch.setattr(
        handoff,
        "_FROZEN_EXPORT_SOURCE_SHA256",
        {relative: handoff._FROZEN_EXPORT_SOURCE_SHA256[relative]},
    )
    with pytest.raises(
        handoff.W8A16ExportHandoffError,
        match="frozen exporter/source closure changed",
    ):
        handoff._verify_frozen_export_source_closure(tmp_path)
