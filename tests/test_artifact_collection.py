from __future__ import annotations

import hashlib
import json

import pytest

from prismaquant.artifact_collection import (
    ArtifactCollectionError,
    CANDIDATE_SCHEMA,
    CATALOG_SCHEMA,
    COST_SNAPSHOT_SCHEMA,
    MODEL_SNAPSHOT_SCHEMA,
    PROBE_CAMPAIGN_SCHEMA,
    TARGET_SCHEMA,
    assignment_byte_breakdown,
    load_record,
    make_candidate,
    make_candidate_catalog,
    make_collection_contract,
    make_collection_manifest,
    make_reference,
    make_stage_receipt,
    make_target_profile,
    reference_for_record,
    verify_record,
    write_record,
)
from prismaquant.artifact_collection_records import make_auxiliary


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reference(label: str, *, size: int = 1, schema: str = "fixture.blob.v1"):
    return make_reference(
        subject_schema=schema,
        subject_id=_digest(f"subject:{label}"),
        content_sha256=_digest(f"content:{label}"),
        size_bytes=size,
    )


def _candidate(
    *,
    rung: int,
    basis_kind: str = "lattice",
    basis_label: str | None = None,
    serialization_label: str = "serialization",
):
    basis_label = basis_label or f"{basis_kind}-{rung}"
    return make_candidate(
        format_semantics={"family": "nvfp4_cb", "rung": rung},
        basis_kind=basis_kind,
        basis_scope="format",
        basis_assets=[_reference(basis_label, size=64)],
        render_contract=_reference("renderer"),
        scale_contract=_reference("scale"),
        activation_contract=_reference("activation"),
        serialization_contract=_reference(serialization_label),
        runtime_contract=_reference("runtime"),
        required_probe_features=["act_sq_sum", "h_trace_raw"],
        required_runtime_features=["gridbook.cb.fp4.product"],
        applicability={"roles": ["linear"]},
        shared_resources=[_reference(basis_label, size=64)],
    )


def test_semantic_id_is_key_order_and_locator_independent():
    subject = _digest("subject")
    first = make_auxiliary(
        "fixture.record",
        {"z": 2, "a": {"right": 1, "left": 0}},
        locators={subject: ["/mnt/a/object.json"]},
    )
    second = make_auxiliary(
        "fixture.record",
        {"a": {"left": 0, "right": 1}, "z": 2},
        locators={subject: ["hf://org/repo/object.json"]},
    )
    assert first["payload_sha256"] == second["payload_sha256"]
    assert reference_for_record(first) == reference_for_record(second)


def test_candidate_id_includes_basis_asset_but_behavior_id_deduplicates_packaging():
    lattice = _candidate(rung=12, basis_kind="lattice", basis_label="lattice-k12")
    learned = _candidate(rung=12, basis_kind="learned", basis_label="learned-k12")
    repacked = _candidate(
        rung=12,
        basis_kind="lattice",
        basis_label="lattice-k12",
        serialization_label="alternate-container",
    )

    assert lattice["payload_sha256"] != learned["payload_sha256"]
    assert lattice["payload"]["behavior_id"] != learned["payload"]["behavior_id"]
    assert lattice["payload_sha256"] != repacked["payload_sha256"]
    assert lattice["payload"]["behavior_id"] == repacked["payload"]["behavior_id"]


def test_catalog_is_an_explicit_set_and_allows_holes_and_family_overlap():
    low = _candidate(rung=1)
    high = _candidate(rung=32)
    learned_high = _candidate(rung=32, basis_kind="learned")
    catalog = make_candidate_catalog([high, learned_high, low])

    assert len(catalog["payload"]["candidates"]) == 3
    assert [row["subject_id"] for row in catalog["payload"]["candidates"]] == sorted(
        row["subject_id"] for row in catalog["payload"]["candidates"]
    )
    assert {row["subject_schema"] for row in catalog["payload"]["candidates"]} == {
        CANDIDATE_SCHEMA
    }


def test_shared_resources_are_charged_once_by_physical_content():
    shared_a = _reference("shared-book", size=64)
    # A second semantic subject may point at the exact same physical content.
    shared_b = make_reference(
        subject_schema="fixture.alias.v1",
        subject_id=_digest("alias-subject"),
        content_sha256=shared_a["content"]["sha256"],
        size_bytes=64,
    )
    fixed = _reference("fixed", size=10)
    result = assignment_byte_breakdown(
        [
            {
                "unit_id": "a",
                "candidate_id": _digest("candidate-a"),
                "local_bytes": 100,
                "shared_resources": [shared_a],
            },
            {
                "unit_id": "b",
                "candidate_id": _digest("candidate-b"),
                "local_bytes": 200,
                "shared_resources": [shared_b],
            },
        ],
        fixed_resources=[fixed],
    )
    assert result["local_bytes"] == 300
    assert result["shared_bytes"] == 74
    assert result["total_bytes"] == 374
    assert len(result["unique_shared_resources"]) == 2

    reversed_result = assignment_byte_breakdown(
        [
            {
                "unit_id": "b",
                "candidate_id": _digest("candidate-b"),
                "local_bytes": 200,
                "shared_resources": [shared_b],
            },
            {
                "unit_id": "a",
                "candidate_id": _digest("candidate-a"),
                "local_bytes": 100,
                "shared_resources": [shared_a],
            },
        ],
        fixed_resources=[fixed],
    )
    assert reversed_result == result


def test_collection_contract_sorts_variants_and_reuses_one_probe():
    common = _reference("common")
    target_8 = make_target_profile(
        artifact_byte_ceiling=5_500_000_000,
        artifact_byte_scope="recursive_package_bytes",
        usable_vram_bytes=8 << 30,
        accounting_rule=common,
        device_profile=_reference("rtx-5060"),
        workload=_reference("chat-8k"),
        placement_constraints={"placement": "gpu_only"},
        fixed_resources=[],
        exclusions=[],
        required_qualification_checks=["structural_inventory"],
    )
    target_16 = make_target_profile(
        artifact_byte_ceiling=13_000_000_000,
        artifact_byte_scope="recursive_package_bytes",
        usable_vram_bytes=16 << 30,
        accounting_rule=common,
        device_profile=_reference("rtx-5080"),
        workload=_reference("chat-8k"),
        placement_constraints={"placement": "gpu_only"},
        fixed_resources=[],
        exclusions=[],
        required_qualification_checks=["structural_inventory"],
    )
    variants = {
        "16gb-balanced": {
            "target_profile": target_16,
            "export_contract": _reference("export-v1"),
        },
        "8gb-lab": {
            "target_profile": target_8,
            "export_contract": _reference("export-v1"),
        },
    }
    contract = make_collection_contract(
        model_snapshot=_reference("model", schema=MODEL_SNAPSHOT_SCHEMA),
        probe_campaign=_reference("probe-once", schema=PROBE_CAMPAIGN_SCHEMA),
        candidate_catalog=_reference("catalog", schema=CATALOG_SCHEMA),
        cost_snapshot=_reference("costs", schema=COST_SNAPSHOT_SCHEMA),
        accounting_rule=common,
        variants=variants,
    )
    reversed_contract = make_collection_contract(
        model_snapshot=_reference("model", schema=MODEL_SNAPSHOT_SCHEMA),
        probe_campaign=_reference("probe-once", schema=PROBE_CAMPAIGN_SCHEMA),
        candidate_catalog=_reference("catalog", schema=CATALOG_SCHEMA),
        cost_snapshot=_reference("costs", schema=COST_SNAPSHOT_SCHEMA),
        accounting_rule=common,
        variants=dict(reversed(list(variants.items()))),
    )
    assert contract["payload_sha256"] == reversed_contract["payload_sha256"]
    assert [row["key"] for row in contract["payload"]["variants"]] == [
        "16gb-balanced",
        "8gb-lab",
    ]


def test_receipts_are_immutable_evidence_not_mutable_status():
    common = _reference("common")
    target = make_target_profile(
        artifact_byte_ceiling=13_000_000_000,
        artifact_byte_scope="recursive_package_bytes",
        usable_vram_bytes=16 << 30,
        accounting_rule=common,
        device_profile=_reference("rtx-5080"),
        workload=_reference("chat-8k"),
        placement_constraints={"placement": "gpu_only"},
        fixed_resources=[],
        exclusions=[],
        required_qualification_checks=["structural_inventory"],
    )
    contract = make_collection_contract(
        model_snapshot=_reference("model", schema=MODEL_SNAPSHOT_SCHEMA),
        probe_campaign=_reference("probe", schema=PROBE_CAMPAIGN_SCHEMA),
        candidate_catalog=_reference("catalog", schema=CATALOG_SCHEMA),
        cost_snapshot=_reference("costs", schema=COST_SNAPSHOT_SCHEMA),
        accounting_rule=common,
        variants={
            "16gb-balanced": {
                "target_profile": target,
                "export_contract": _reference("export-v1"),
            }
        },
    )
    producer = _reference("producer")
    receipt = make_stage_receipt(
        collection_contract=reference_for_record(contract),
        variant_key="16gb-balanced",
        stage="export",
        outcome="accepted",
        inputs=[_reference("solve")],
        outputs=[_reference("artifact")],
        evidence=[_reference("inventory")],
        producer=producer,
    )
    manifest = make_collection_manifest(
        collection_contract=contract,
        receipts=[receipt],
    )
    assert "status" not in receipt["payload"]
    assert manifest["payload"]["receipts"][0]["subject_id"] == receipt["payload_sha256"]


def test_strict_envelope_refuses_tamper_unknown_fields_and_nonfinite_values():
    record = make_auxiliary("fixture.record", {"value": 1})
    tampered = json.loads(json.dumps(record))
    tampered["payload"]["data"]["value"] = 2
    with pytest.raises(ArtifactCollectionError, match="payload_sha256"):
        verify_record(tampered)

    extra = dict(record, status="complete")
    with pytest.raises(ArtifactCollectionError, match="field set differs"):
        verify_record(extra)

    with pytest.raises(ArtifactCollectionError, match="finite canonical JSON"):
        make_auxiliary("fixture.record", {"value": float("nan")})


def test_loader_refuses_duplicate_json_members_and_nonfinite_values(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"x","schema":"y","payload":{},"payload_sha256":"'
        + "0" * 64
        + '"}',
        encoding="utf-8",
    )
    with pytest.raises(ArtifactCollectionError, match="duplicate member"):
        load_record(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ArtifactCollectionError, match="non-finite"):
        load_record(nonfinite)


def test_target_rejects_bool_or_zero_byte_limits():
    common = _reference("common")
    with pytest.raises(ArtifactCollectionError, match="expected an integer"):
        make_target_profile(
            artifact_byte_ceiling=True,
            artifact_byte_scope="recursive_package_bytes",
            usable_vram_bytes=1,
            accounting_rule=common,
            device_profile=common,
            workload=common,
            placement_constraints={"placement": "gpu_only"},
            fixed_resources=[],
            exclusions=[],
            required_qualification_checks=["structural_inventory"],
        )
    with pytest.raises(ArtifactCollectionError, match="positive"):
        make_target_profile(
            artifact_byte_ceiling=0,
            artifact_byte_scope="recursive_package_bytes",
            usable_vram_bytes=1,
            accounting_rule=common,
            device_profile=common,
            workload=common,
            placement_constraints={"placement": "gpu_only"},
            fixed_resources=[],
            exclusions=[],
            required_qualification_checks=["structural_inventory"],
        )


def test_record_publication_is_round_trip_and_no_clobber(tmp_path):
    record = make_auxiliary("fixture.record", {"value": 1})
    destination = tmp_path / "record.json"
    write_record(destination, record)
    assert load_record(destination) == record
    with pytest.raises(ArtifactCollectionError, match="will not be replaced"):
        write_record(destination, record)

    symlink = tmp_path / "link.json"
    symlink.symlink_to(destination)
    with pytest.raises(ArtifactCollectionError, match="will not be replaced"):
        write_record(symlink, record)


def test_locator_overlay_is_not_written_and_reference_matches_published_bytes(tmp_path):
    subject = _digest("located-subject")
    record = make_auxiliary(
        "fixture.located",
        {"value": 1},
        locators={subject: ["/staging/record.json"]},
    )
    reference = reference_for_record(record)
    destination = tmp_path / "record.json"
    write_record(destination, record)
    encoded = destination.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == reference["content"]["sha256"]
    assert len(encoded) == reference["content"]["size_bytes"]
    assert "locators" not in load_record(destination)


def test_same_semantic_reference_cannot_equivocate_on_content():
    first = _reference("first")
    conflicting = make_reference(
        subject_schema=first["subject_schema"],
        subject_id=first["subject_id"],
        content_sha256=_digest("different-content"),
        size_bytes=2,
    )
    with pytest.raises(ArtifactCollectionError, match="equivocation|conflicting content"):
        make_candidate(
            format_semantics={"family": "nvfp4_cb", "rung": 1},
            basis_kind="lattice",
            basis_scope="format",
            basis_assets=sorted([first, conflicting], key=lambda row: row["content"]["sha256"]),
            render_contract=_reference("renderer"),
            scale_contract=_reference("scale"),
            activation_contract=None,
            serialization_contract=_reference("serialization"),
            runtime_contract=_reference("runtime"),
            required_probe_features=[],
            required_runtime_features=[],
            applicability={"roles": ["linear"]},
            shared_resources=[],
        )


def test_collection_rejects_target_with_a_different_accounting_rule():
    common = _reference("accounting-a")
    target = make_target_profile(
        artifact_byte_ceiling=1,
        artifact_byte_scope="recursive_package_bytes",
        usable_vram_bytes=1,
        accounting_rule=_reference("accounting-b"),
        device_profile=_reference("device"),
        workload=_reference("workload"),
        placement_constraints={"placement": "gpu_only"},
        fixed_resources=[],
        exclusions=[],
        required_qualification_checks=["structural_inventory"],
    )
    with pytest.raises(ArtifactCollectionError, match="differs from the collection"):
        make_collection_contract(
            model_snapshot=_reference("model"),
            probe_campaign=_reference("probe"),
            candidate_catalog=_reference("catalog"),
            cost_snapshot=_reference("costs"),
            accounting_rule=common,
            variants={
                "variant": {
                    "target_profile": target,
                    "export_contract": _reference("export"),
                }
            },
        )


def test_text_sequence_constructor_rejects_a_bare_string():
    with pytest.raises(ArtifactCollectionError, match="array of strings"):
        make_candidate(
            format_semantics={"family": "nvfp4_cb", "rung": 1},
            basis_kind="lattice",
            basis_scope="format",
            basis_assets=[],
            render_contract=_reference("renderer"),
            scale_contract=_reference("scale"),
            activation_contract=None,
            serialization_contract=_reference("serialization"),
            runtime_contract=_reference("runtime"),
            required_probe_features="hessian",
            required_runtime_features=[],
            applicability={"roles": ["linear"]},
            shared_resources=[],
        )


def test_manifest_refuses_receipt_for_another_contract_or_unknown_variant():
    common = _reference("common")
    target = make_target_profile(
        artifact_byte_ceiling=1,
        artifact_byte_scope="recursive_package_bytes",
        usable_vram_bytes=1,
        accounting_rule=common,
        device_profile=_reference("device"),
        workload=_reference("workload"),
        placement_constraints={"placement": "gpu_only"},
        fixed_resources=[],
        exclusions=[],
        required_qualification_checks=["structural_inventory"],
    )
    contract = make_collection_contract(
        model_snapshot=_reference("model", schema=MODEL_SNAPSHOT_SCHEMA),
        probe_campaign=_reference("probe", schema=PROBE_CAMPAIGN_SCHEMA),
        candidate_catalog=_reference("catalog", schema=CATALOG_SCHEMA),
        cost_snapshot=_reference("cost", schema=COST_SNAPSHOT_SCHEMA),
        accounting_rule=common,
        variants={
            "known": {
                "target_profile": target,
                "export_contract": _reference("export"),
            }
        },
    )
    other_contract = make_reference(
        subject_schema="prismaquant.artifact_collection.contract.v1",
        subject_id=_digest("other-contract"),
        content_sha256=_digest("other-contract-content"),
        size_bytes=1,
    )
    receipt = make_stage_receipt(
        collection_contract=other_contract,
        variant_key="known",
        stage="solve",
        outcome="accepted",
        inputs=[_reference("cost")],
        outputs=[_reference("assignment")],
        evidence=[_reference("solver-receipt")],
        producer=_reference("producer"),
    )
    with pytest.raises(ArtifactCollectionError, match="does not bind this"):
        make_collection_manifest(collection_contract=contract, receipts=[receipt])


def test_candidate_unknown_identity_field_is_rejected():
    candidate = _candidate(rung=4)
    candidate["payload"]["display_name"] = "friendly alias"
    # Rehashing cannot smuggle a new identity-bearing field through the schema.
    candidate["payload_sha256"] = hashlib.sha256(
        json.dumps(
            candidate["payload"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(ArtifactCollectionError, match="field set differs"):
        verify_record(candidate)
