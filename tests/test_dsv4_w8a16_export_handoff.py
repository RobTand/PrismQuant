from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from prismaquant.cb_anchored_cost import write_exportable_artifacts
from prismaquant.nvfp4_cb_footprint import assignment_serialization_sha256
import prismaquant.dsv4_w8a16_export_handoff as handoff
import prismaquant.dsv4_w8a16_legacy_compat as legacy_compat


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
                "codebook_source_by_format": {
                    "FP8_CB_K28": "learned",
                    "FP8_CB_K36": "learned",
                    "FP8_CB_K44": "learned",
                    "FP8_CB_K48": "lattice",
                    "NVFP4_CB_K16": "lattice",
                    "NVFP4_CB_K18": "lattice",
                },
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
    col.write_bytes(pickle.dumps({"fixture": [1.0, 2.0]}))
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
        "model.layers.2.self_attn.wq_a": "FP8_BLOCK_UE8M0_SOURCE",
        "model.layers.18.mlp.experts.0.down_proj": "FP8_CB_K28",
        "model.layers.0.self_attn.wq_b": "FP8_CB_K36",
        "model.layers.3.self_attn.wq_b": "FP8_CB_K44",
        "model.layers.4.self_attn.wq_b": "FP8_CB_K48",
        "model.layers.5.mlp.shared_experts.down_proj": "NVFP4_CB_K16",
        "model.layers.6.mlp.shared_experts.down_proj": "NVFP4_CB_K18",
    }
    assignment_sha256 = assignment_serialization_sha256(assignment)
    monkeypatch.setattr(handoff, "DSV4_TOTAL_UNITS", len(assignment))
    monkeypatch.setattr(
        handoff, "DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256", assignment_sha256
    )
    monkeypatch.setattr(
        handoff,
        "_LEGACY_W8A16_FORMAT_COUNTS",
        {
            "FP8_BLOCK_UE8M0_SOURCE": 1,
            "FP8_CB_K28": 1,
            "FP8_CB_K36": 1,
            "FP8_CB_K44": 1,
            "FP8_CB_K48": 1,
            "NVFP4_CB_K16": 1,
            "NVFP4_CB_K18": 1,
        },
    )
    monkeypatch.setattr(
        handoff,
        "_LEGACY_W8A16_ROUTED_K28_QNAMES",
        frozenset({"model.layers.18.mlp.experts.0.down_proj"}),
    )
    monkeypatch.setattr(
        handoff,
        "_LEGACY_W8A16_DENSE_K36_QNAMES",
        frozenset({"model.layers.0.self_attn.wq_b"}),
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
        **handoff._LEGACY_GRIDBOOK_RUNTIME,
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
    publication_manifest = json.loads(
        (publication / ".anchored_publish.json").read_text()
    )
    monkeypatch.setattr(
        handoff,
        "_LEGACY_W8A16_PUBLICATION_IDENTITY_SHA256",
        publication_manifest["identity_sha256"],
    )
    monkeypatch.setattr(
        handoff,
        "_LEGACY_W8A16_PUBLISHED_SHA256",
        {
            name: _sha256(publication / name)
            for name in handoff._PUBLISHED_FILES
        },
    )

    source = tmp_path / "source"
    source.mkdir()
    source_identity = tmp_path / "source-identity.json"
    source_identity.write_text("{}")
    bundle = tmp_path / "bundle.pqcb"
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(
        handoff,
        "_LEGACY_W8A16_SOURCE_IDENTITY_FILE_SHA256",
        _sha256(source_identity),
    )
    monkeypatch.setattr(
        handoff, "_LEGACY_W8A16_SOURCE_CONTENT_SHA256", "b" * 64,
    )
    monkeypatch.setattr(handoff, "_LEGACY_W8A16_SOURCE_SHARD_COUNT", 1)
    monkeypatch.setattr(
        handoff, "_LEGACY_W8A16_BUNDLE_FILE_SHA256", _sha256(bundle),
    )
    monkeypatch.setattr(
        handoff, "_LEGACY_W8A16_BUNDLE_CONTENT_SHA256", "d" * 64,
    )
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
        "prismaquant.cost_streaming.compact_streamed_model_identity",
        lambda _identity, **_kwargs: {
            "schema": "prismaquant.streamed_model_identity.v1",
            "content_sha256": "b" * 64,
            "resolved_commit": "fixture",
            "checkpoint_shards": 1,
            "checkpoint_tensors": 1,
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
    monkeypatch.setattr(
        handoff,
        "_verify_frozen_export_source_closure",
        lambda _root: {
            "schema": handoff.DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA,
            "file_count": 3,
            "tree_sha256": "e" * 64,
            "pin_manifest_sha256": "f" * 64,
            "pin_identity_sha256": "1" * 64,
            "identity_sha256": "2" * 64,
        },
    )
    return {
        "publication": publication,
        "raw": raw,
        "source": source,
        "source_identity": source_identity,
        "bundle": bundle,
        "assignment": assignment,
        "assignment_sha256": assignment_sha256,
    }


def _verify(tmp_path: Path, fixture):
    return handoff.verify_dsv4_w8a16_export_handoff(
        publication_dir=fixture["publication"],
        approved_raw_publication_dir=fixture["raw"],
        source_model_dir=fixture["source"],
        source_identity_path=fixture["source_identity"],
        codebook_bundle_path=fixture["bundle"],
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
    assert receipt["unit_count"] == 7
    assert receipt["fp8_block_w8a16_count"] == 1
    assert receipt["legacy_compatibility"]["format_counts"] == (
        handoff._LEGACY_W8A16_FORMAT_COUNTS
    )
    assert receipt["source_checkpoint"]["content_sha256"] == "b" * 64
    closure = receipt["frozen_export_source_closure"]
    assert closure["schema"] == (
        handoff.DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA
    )
    assert closure["tree_sha256"] == "e" * 64
    assert "frozen_export_code_sha256" not in receipt
    assert receipt["output_absent"] is True
    assert not (tmp_path / "artifact").exists()


def test_handoff_refuses_publication_byte_drift(monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    publication = fixture["publication"]
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


@pytest.mark.parametrize("substitution", ["publication-directory", "child"])
def test_handoff_refuses_publication_symlink_substitution(
    monkeypatch,
    tmp_path: Path,
    substitution: str,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    if substitution == "publication-directory":
        alias = tmp_path / "publication-alias"
        alias.symlink_to(fixture["publication"], target_is_directory=True)
        fixture["publication"] = alias
    else:
        selection = Path(fixture["publication"]) / "selection.json"
        target = Path(fixture["raw"]) / "selection.json"
        selection.unlink()
        selection.symlink_to(target)

    with pytest.raises(
        handoff.W8A16ExportHandoffError,
        match="real directory|regular file",
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


def test_bundle_parser_consumes_the_exact_held_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import prismaquant.cb_learned_bundle as bundle_module

    path = tmp_path / "bundle.pqcb"
    original = b"exact reviewed bundle bytes"
    path.write_bytes(original)
    sources = {"FP8_CB_K28": "learned"}
    digests = {"fixture": "a" * 64}
    monkeypatch.setattr(
        handoff,
        "cb_serialization_metadata_from_assignment_payload",
        lambda _payload: ({
            "codebook_content_sha256": digests,
            "codebook_source_by_format": sources,
        }, {}),
    )

    def _load_snapshot(snapshot_path):
        assert Path(snapshot_path).read_bytes() == original
        path.write_bytes(b"substituted after held read")
        return SimpleNamespace(
            codebook_content_digests=digests,
            codebook_source_by_format=sources,
            bundle_content_sha256="b" * 64,
        )

    monkeypatch.setattr(bundle_module, "load_bundle", _load_snapshot)

    observed = handoff._verify_bundle(path, {})

    assert observed["file_sha256"] == hashlib.sha256(original).hexdigest()
    assert observed["bundle_content_sha256"] == "b" * 64
    assert path.read_bytes() == b"substituted after held read"


def test_source_identity_validator_consumes_the_exact_held_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    identity_path = Path(fixture["source_identity"])
    original = identity_path.read_bytes()

    def _validate(source_path, snapshot_path, **_kwargs):
        assert str(source_path).startswith("/proc/self/fd/")
        assert Path(snapshot_path).read_bytes() == original
        identity_path.write_bytes(b"substituted after held read")
        return {
            "content_sha256": "b" * 64,
            "shards": [{"sha256": "c" * 64}],
        }

    monkeypatch.setattr(
        "prismaquant.cost_streaming.validate_cached_streamed_model_identity",
        _validate,
    )

    receipt = _verify(tmp_path, fixture)

    assert receipt["source_checkpoint"]["identity_file_sha256"] == (
        hashlib.sha256(original).hexdigest()
    )
    assert identity_path.read_bytes() == b"substituted after held read"


def _write_runtime_pin(root: Path) -> dict[str, str]:
    package = root / "prismaquant"
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file()
        and path.name != handoff._SOURCE_CLOSURE_PIN_NAME
        and "__pycache__" not in path.parts
    }
    pin: dict[str, object] = {
        "schema": handoff.DSV4_W8A16_EXPORT_SOURCE_CLOSURE_PIN_SCHEMA,
        "files_sha256": files,
    }
    pin["identity_sha256"] = handoff.canonical_json_sha256(
        pin, where="test W8A16 runtime pin",
    )
    (package / handoff._SOURCE_CLOSURE_PIN_NAME).write_bytes(
        handoff.canonical_json_bytes(pin) + b"\n"
    )
    return files


def _closure_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    package = tmp_path / "prismaquant"
    (package / "model_profiles/specs").mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"# fixture\n")
    (package / "exporter.py").write_bytes(b"VALUE = 1\n")
    (package / "model_profiles/specs/model.json").write_bytes(b"{}\n")
    return package, _write_runtime_pin(tmp_path)


def test_complete_runtime_closure_binds_every_regular_package_file(tmp_path):
    _package, files = _closure_fixture(tmp_path)

    observed = handoff._verify_frozen_export_source_closure(tmp_path)

    assert observed["schema"] == (
        handoff.DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA
    )
    assert observed["file_count"] == len(files)
    assert observed["tree_sha256"] == handoff.canonical_json_sha256(
        files, where="test complete runtime ledger",
    )
    assert "files_sha256" not in observed


def test_project_runtime_closure_pin_matches_the_complete_current_tree() -> None:
    root = Path(__file__).resolve().parents[1]

    observed = handoff._verify_frozen_export_source_closure(root)

    assert observed == {
        "schema": "prismaquant.dsv4_w8a16.export_source_closure.v2",
        "file_count": 229,
        "tree_sha256": (
            "050ab3b5d3d03305835e41b80c678169f49238f052067e565351366d23c1ef37"
        ),
        "pin_manifest_sha256": (
            "f53e9f6b470c652d607829133d23ca2d9876d6d0d82b138c2b5c330d8d15289a"
        ),
        "pin_identity_sha256": (
            "7082fa2f56f60539131eb4e56fbc66f4efb1fde7b5a2818a850ec97f7c864c87"
        ),
        "identity_sha256": (
            "3b077bf5f7976d5d25d6dbb229e0b095ab5c3f51b96c0e56c2c7fef967dbd41e"
        ),
    }


@pytest.mark.parametrize("mutation", ["drift", "extra", "missing"])
def test_complete_runtime_closure_refuses_exact_ledger_changes(
    tmp_path: Path,
    mutation: str,
) -> None:
    package, _files = _closure_fixture(tmp_path)
    if mutation == "drift":
        (package / "exporter.py").write_bytes(b"VALUE = 2\n")
    elif mutation == "extra":
        (package / "extra.py").write_bytes(b"EXTRA = True\n")
    else:
        (package / "exporter.py").unlink()

    with pytest.raises(
        handoff.W8A16ExportHandoffError,
        match="runtime closure changed",
    ):
        handoff._verify_frozen_export_source_closure(tmp_path)


@pytest.mark.parametrize("mutation", ["symlink", "special", "bytecode", "temp"])
def test_complete_runtime_closure_rejects_untracked_runtime_entry_kinds(
    tmp_path: Path,
    mutation: str,
) -> None:
    package, _files = _closure_fixture(tmp_path)
    if mutation == "symlink":
        (package / "alias.py").symlink_to(package / "exporter.py")
        message = "symlink"
    elif mutation == "special":
        os.mkfifo(package / "channel")
        message = "special file"
    elif mutation == "bytecode":
        (package / "untracked.pyc").write_bytes(b"bytecode")
        message = "bytecode"
    else:
        (package / ".tmp-race").mkdir()
        message = "temporary/control"

    with pytest.raises(handoff.W8A16ExportHandoffError, match=message):
        handoff._verify_frozen_export_source_closure(tmp_path)


@pytest.mark.parametrize(
    ("old_qname", "new_qname"),
    [
        (
            "model.layers.18.mlp.experts.0.down_proj",
            "model.layers.7.self_attn.wq_b",
        ),
        (
            "model.layers.0.self_attn.wq_b",
            "model.layers.18.mlp.experts.1.down_proj",
        ),
    ],
)
def test_legacy_ledger_refuses_same_count_qname_swaps(
    monkeypatch,
    tmp_path: Path,
    old_qname: str,
    new_qname: str,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    assignment = dict(fixture["assignment"])
    assignment[new_qname] = assignment.pop(old_qname)

    with pytest.raises(handoff.W8A16ExportHandoffError, match="cells differ"):
        handoff.legacy_w8a16_assignment_compatibility(assignment)


@pytest.mark.parametrize("extra_format", ["FP8_CB_K32", "FP8_CB_K35"])
def test_legacy_ledger_refuses_unapproved_reader_or_research_rungs(
    monkeypatch,
    tmp_path: Path,
    extra_format: str,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    assignment = dict(fixture["assignment"])
    assignment["model.layers.3.self_attn.wq_b"] = extra_format

    with pytest.raises(handoff.W8A16ExportHandoffError, match="format counts"):
        handoff.legacy_w8a16_assignment_compatibility(assignment)


def test_real_sealed_w8a16_assignment_has_exact_33325_unit_ledger() -> None:
    path = Path(
        "/home/rob/dq-runs/dsv4-flash-0731/"
        "aura-cb-reprice-streamed-cached/artifacts/"
        "exportable-aura-w8a16-readmission-packed-alias-v2/layer_config.json"
    )
    if not path.is_file():
        pytest.skip("local sealed DSv4 W8A16 publication is not mounted")
    from prismaquant.layer_config import load_assignment

    assignment = load_assignment(path)
    ledger = handoff.legacy_w8a16_assignment_compatibility(assignment)

    assert len(assignment) == 33_325
    assert ledger["format_counts"] == handoff._LEGACY_W8A16_FORMAT_COUNTS
    assert ledger["exception_map"]["routed_fp8_cb_k28"]["count"] == 6_144
    assert ledger["exception_map"]["dense_fp8_cb_k36"]["count"] == 3
    assert ledger["identity_sha256"] == (
        "77b1b50b1cd6a24bf44624a3bc05b0c68b3a81b0baf86466116fdb0b373b1a87"
    )


def _derive_compatibility(
    tmp_path: Path,
    fixture: dict[str, object],
    receipt: dict[str, object],
    **overrides,
):
    arguments = {
        "model_dir": fixture["source"],
        "layer_config_path": Path(fixture["publication"]) / "layer_config.json",
        "out_dir": tmp_path / "artifact",
        "col_weights": {"fixture": [1.0, 2.0]},
        "col_weights_path": (
            Path(fixture["publication"]) / "cb_col_weights.pkl"
        ),
        "codebook_bundle_path": fixture["bundle"],
        "repo_root": Path(__file__).resolve().parents[1],
    }
    arguments.update(overrides)
    return legacy_compat.derive_dsv4_w8a16_legacy_compatibility(
        receipt, **arguments,
    )


def test_receipt_replay_derives_only_representative_exception_cells(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _verify(tmp_path, fixture)

    capability = _derive_compatibility(tmp_path, fixture, receipt)
    bundle_path = Path(fixture["bundle"])
    original_bundle = bundle_path.read_bytes()

    assert capability.allows(
        "model.layers.18.mlp.experts.0.down_proj", "FP8_CB_K28"
    )
    assert capability.allows(
        "model.layers.0.self_attn.wq_b", "FP8_CB_K36"
    )
    assert not capability.allows(
        "model.layers.3.self_attn.wq_b", "FP8_CB_K44"
    )
    assert not capability.allows(
        "model.layers.7.self_attn.wq_b", "FP8_CB_K28"
    )
    stamp = capability.stamp()
    assert stamp["exception_map"]["routed_fp8_cb_k28"]["count"] == 1
    assert stamp["exception_map"]["dense_fp8_cb_k36"]["count"] == 1
    assert stamp["layer_config_file_sha256"] == hashlib.sha256(
        (Path(fixture["publication"]) / "layer_config.json").read_bytes()
    ).hexdigest()
    assert stamp["source_content_sha256"] == "b" * 64
    assert stamp["source_model_identity"]["checkpoint_shards"] == 1
    assert stamp["codebook_bundle_file_sha256"] == hashlib.sha256(
        original_bundle
    ).hexdigest()
    assert stamp["runtime_closure_identity_sha256"] == "2" * 64
    with capability.open_bound_codebook_bundle() as snapshot_path:
        bundle_path.write_bytes(b"changed after capability derivation")
        assert Path(snapshot_path).read_bytes() == original_bundle


def test_canonical_receipt_file_replays_and_noncanonical_or_symlink_refuses(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _verify(tmp_path, fixture)
    receipt_path = tmp_path / "handoff.json"
    receipt_path.write_bytes(handoff.canonical_json_bytes(receipt) + b"\n")

    capability = _derive_compatibility(tmp_path, fixture, receipt_path)
    assert capability.receipt_identity_sha256 == receipt["identity_sha256"]

    receipt_path.write_bytes(handoff.canonical_json_bytes(receipt))
    with pytest.raises(
        legacy_compat.W8A16LegacyCompatibilityError,
        match="canonical JSON file encoding",
    ):
        _derive_compatibility(tmp_path, fixture, receipt_path)

    receipt_path.write_bytes(handoff.canonical_json_bytes(receipt) + b"\n")
    alias = tmp_path / "handoff-alias.json"
    alias.symlink_to(receipt_path)
    with pytest.raises(
        legacy_compat.W8A16LegacyCompatibilityError,
        match="receipt is unreadable",
    ):
        _derive_compatibility(tmp_path, fixture, alias)


def test_capability_refuses_layer_bytes_changed_after_receipt_replay(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _verify(tmp_path, fixture)
    capability = _derive_compatibility(tmp_path, fixture, receipt)
    layer_path = Path(fixture["publication"]) / "layer_config.json"

    with layer_path.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(
        legacy_compat.W8A16LegacyCompatibilityError,
        match="bytes changed after handoff replay",
    ):
        capability.read_bound_layer_config(layer_path)
    assert not (tmp_path / "artifact").exists()


def test_exact_routed_stack_capability_requires_full_expected_member_mapping() -> None:
    routed = handoff._LEGACY_W8A16_ROUTED_K28_QNAMES
    capability = legacy_compat.DSV4W8A16LegacyCompatibility(
        receipt_identity_sha256="a" * 64,
        publication_identity_sha256="1" * 64,
        assignment_sha256="b" * 64,
        output_path="/tmp/never",
        layer_config_path="/tmp/layer.json",
        layer_config_file_sha256="c" * 64,
        source_identity_file_sha256="2" * 64,
        source_content_sha256="3" * 64,
        source_model_identity={"content_sha256": "3" * 64},
        codebook_bundle_file_sha256="4" * 64,
        codebook_bundle_content_sha256="5" * 64,
        runtime_pin_sha256="6" * 64,
        runtime_closure_identity_sha256="7" * 64,
        col_weights_content_sha256="d" * 64,
        ledger={"exception_map": {}},
        _routed_k28_qnames=routed,
        _dense_k36_qnames=handoff._LEGACY_W8A16_DENSE_K36_QNAMES,
        _codebook_bundle_payload=b"fixture",
    )
    down_members = {
        ("down_proj", expert): (
            f"model.layers.18.mlp.experts.{expert}.down_proj"
        )
        for expert in range(256)
    }

    assert capability.allows_group(
        "model.layers.18.mlp.experts.down_proj",
        "FP8_CB_K28",
        down_members,
    )
    assert not capability.allows_group(
        "model.layers.17.mlp.experts.down_proj",
        "FP8_CB_K28",
        down_members,
    )
    assert not capability.allows_group(
        "model.layers.18.mlp.experts.down_proj",
        "FP8_CB_K28",
        dict(list(down_members.items())[:-1]),
    )
    crossed = dict(down_members)
    crossed[("down_proj", 0)] = "model.layers.19.mlp.experts.0.down_proj"
    assert not capability.allows_group(
        "model.layers.18.mlp.experts.down_proj",
        "FP8_CB_K28",
        crossed,
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("subset_prefixes", ["model.layers.0."]),
        ("reuse_prior", "/tmp/prior"),
        ("per_expert_config_path", "/tmp/per-expert.json"),
        ("dspark_cb_sidecar", True),
        ("exclude_namespaces", ["mtp."]),
    ],
)
def test_compatibility_refuses_partial_or_reuse_export_modes(
    monkeypatch,
    tmp_path: Path,
    argument: str,
    value: object,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _verify(tmp_path, fixture)

    with pytest.raises(
        legacy_compat.W8A16LegacyCompatibilityError,
        match="forbids subset, reuse, per-expert, DSpark",
    ):
        _derive_compatibility(tmp_path, fixture, receipt, **{argument: value})


@pytest.mark.parametrize(
    "argument",
    [
        "model_dir",
        "layer_config_path",
        "out_dir",
        "col_weights_path",
        "codebook_bundle_path",
    ],
)
def test_compatibility_refuses_every_cross_assignment_path(
    monkeypatch,
    tmp_path: Path,
    argument: str,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _verify(tmp_path, fixture)
    alternate = tmp_path / f"alternate-{argument}"
    if argument in {"model_dir", "out_dir"}:
        if argument == "model_dir":
            alternate.mkdir()
    else:
        alternate.write_bytes(b"other")

    with pytest.raises(
        legacy_compat.W8A16LegacyCompatibilityError,
        match="cross or differ",
    ):
        _derive_compatibility(
            tmp_path, fixture, receipt, **{argument: alternate},
        )


def test_compatibility_replays_forged_self_rehashed_receipt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _verify(tmp_path, fixture)
    forged = json.loads(json.dumps(receipt))
    forged["selection"]["chosen_achieved_bits"] = 9.0
    forged["identity_sha256"] = handoff.canonical_json_sha256(
        {key: value for key, value in forged.items() if key != "identity_sha256"},
        where="forged W8A16 handoff receipt",
    )

    with pytest.raises(
        legacy_compat.W8A16LegacyCompatibilityError,
        match="differs from independently replayed facts",
    ):
        _derive_compatibility(tmp_path, fixture, forged)


def test_compatibility_refuses_exporter_column_weight_value_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _verify(tmp_path, fixture)

    with pytest.raises(
        legacy_compat.W8A16LegacyCompatibilityError,
        match="column weights differ",
    ):
        _derive_compatibility(
            tmp_path,
            fixture,
            receipt,
            col_weights={"fixture": [1.0, 3.0]},
        )
