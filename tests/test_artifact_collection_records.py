from __future__ import annotations

import copy
import hashlib

import pytest

from prismaquant.artifact_collection import (
    ArtifactCollectionError,
    CONTRACT_SCHEMA,
    EXPORT_SCHEMA,
    MANIFEST_SCHEMA,
    make_candidate,
    make_candidate_catalog,
    make_collection_contract,
    make_collection_manifest,
    make_reference,
    make_stage_receipt,
    make_target_profile,
    reference_for_record,
    seal_record,
    write_record,
)
from prismaquant.artifact_collection_cli import main as collection_cli
from prismaquant.artifact_collection_records import (
    make_auxiliary,
    make_cost_snapshot,
    make_device_qualification,
    make_export_record,
    make_market_snapshot,
    make_model_snapshot,
    make_probe_campaign,
    make_qualification_evidence,
    make_release_decision,
    make_solve,
    make_unit_ledger,
    verify_collection_graph,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reference(label: str, *, size: int = 1) -> dict[str, object]:
    return make_reference(
        subject_schema="fixture.blob.v1",
        subject_id=_digest(f"subject:{label}"),
        content_sha256=_digest(f"content:{label}"),
        size_bytes=size,
    )


def _collection_records(
    *,
    probe_features: tuple[str, ...] = ("act_sq_sum",),
    include_assignment: bool = True,
    assignment_local_bytes: int = 128,
    assignment_resources: tuple[dict[str, object], ...] | None = None,
    cost_status: str = "measured",
    omit_cost_cell: bool = False,
    qualification_device: dict[str, object] | None = None,
    check_outcome: str = "passed",
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    producer = _reference("producer")
    accounting = _reference("whole-artifact-accounting")
    device = _reference("rtx-5060")
    workload = _reference("chat-8k")
    placement = {"mode": "single_gpu", "offload": "forbidden"}
    runtime = _reference("runtime-v11")
    lattice = _reference("nvfp4-k12-lattice", size=32)
    candidate = make_candidate(
        format_semantics={"family": "nvfp4_cb", "rung": 12},
        basis_kind="lattice",
        basis_scope="format",
        basis_assets=[lattice],
        render_contract=_reference("renderer"),
        scale_contract=_reference("scale-v2"),
        activation_contract=_reference("activation"),
        serialization_contract=_reference("serialization"),
        runtime_contract=runtime,
        required_probe_features=["act_sq_sum"],
        required_runtime_features=["gridbook.cb.fp4.product"],
        applicability={"roles": ["linear"]},
        shared_resources=[lattice],
    )
    catalog = make_candidate_catalog([candidate])
    ledger = make_unit_ledger(
        model_content_sha256=_digest("model-content"),
        units=[
            {
                "unit_id": "body.0.linear",
                "qname": "model.layers.0.linear.weight",
                "shape": [2, 2],
                "parameter_count": 4,
                "disposition": "assign",
            },
            {
                "unit_id": "lm_head",
                "qname": "lm_head.weight",
                "shape": [1, 2],
                "parameter_count": 2,
                "disposition": "exclude",
            },
        ],
    )
    source_content = make_reference(
        subject_schema="fixture.blob.v1",
        subject_id=_digest("subject:source-model"),
        content_sha256=_digest("model-content"),
        size_bytes=12_000,
    )
    model = make_model_snapshot(
        source_content=source_content,
        model_profile=_reference("qwen3.8-27b-profile"),
        unit_ledger=ledger,
        source_parameter_count=6,
        source_tree_sha256=_digest("producer-tree"),
        producer=producer,
    )
    probe = make_probe_campaign(
        model_snapshot=reference_for_record(model),
        probe_blob=_reference("probe-blob", size=400),
        calibration=_reference("calibration", size=200),
        token_content=_reference("tokens", size=100),
        measured_features=list(probe_features),
        covered_unit_ids=["body.0.linear"],
        missing_unit_ids=[],
        merge_receipt=_reference("probe-merge"),
        producer=producer,
    )
    metrics = (
        {"local_bytes": 128, "weighted_mse": 0.125}
        if cost_status != "unavailable"
        else {}
    )
    cost_cells = [] if omit_cost_cell else [
        {
            "unit_id": "body.0.linear",
            "candidate_id": candidate["payload_sha256"],
            "status": cost_status,
            "metrics": metrics,
            "anchors": [],
        }
    ]
    cost = make_cost_snapshot(
        model_snapshot=reference_for_record(model),
        probe_campaign=reference_for_record(probe),
        candidate_catalog=reference_for_record(catalog),
        observations=_reference("cost-observations", size=80),
        metric_contracts={"weighted_mse": {"direction": "minimize"}},
        accounting_rule=accounting,
        cells=cost_cells,
        producer=producer,
    )
    target = make_target_profile(
        artifact_byte_ceiling=1_024,
        artifact_byte_scope="recursive_package_bytes",
        usable_vram_bytes=8 << 30,
        accounting_rule=accounting,
        device_profile=device,
        workload=workload,
        placement_constraints=placement,
        fixed_resources=[],
        exclusions=["w8a16"],
        required_qualification_checks=["numerical_parity"],
    )
    contract = make_collection_contract(
        model_snapshot=reference_for_record(model),
        probe_campaign=reference_for_record(probe),
        candidate_catalog=reference_for_record(catalog),
        cost_snapshot=reference_for_record(cost),
        accounting_rule=accounting,
        variants={
            "rtx5060-8gb": {
                "target_profile": target,
                "export_contract": _reference("export-contract-v1"),
            }
        },
    )
    assignments = []
    if include_assignment:
        assignments.append(
            {
                "unit_id": "body.0.linear",
                "candidate_id": candidate["payload_sha256"],
                "partition": "optimized",
                "local_bytes": assignment_local_bytes,
                "shared_resources": (
                    list(assignment_resources)
                    if assignment_resources is not None
                    else [lattice]
                ),
            }
        )
    solve = make_solve(
        model_snapshot=reference_for_record(model),
        probe_campaign=reference_for_record(probe),
        candidate_catalog=reference_for_record(catalog),
        cost_snapshot=reference_for_record(cost),
        target_profile=reference_for_record(target),
        assignments=assignments,
        fixed_resources=[],
        solver={"name": "aqua", "version": 1},
        predicted_metrics={"weighted_mse": 0.125},
        producer=producer,
    )
    export = make_export_record(
        solve=solve,
        target_profile=reference_for_record(target),
        artifact=_reference("export-root", size=256),
        files=[
            {
                "path": "model.safetensors",
                "size_bytes": 256,
                "sha256": _digest("model.safetensors"),
            }
        ],
        tensors=[
            {
                "unit_id": "body.0.linear",
                "name": "model.layers.0.linear.weight",
                "file": "model.safetensors",
                "dtype": "uint8",
                "shape": [128],
                "sha256": _digest("linear-tensor"),
            },
            {
                "unit_id": "lm_head",
                "name": "lm_head.weight",
                "file": "model.safetensors",
                "dtype": "bfloat16",
                "shape": [1, 2],
                "sha256": _digest("lm-head-tensor"),
            }
        ],
        codebooks=[lattice],
        runtime_artifact=runtime,
        byte_scope="recursive_package_bytes",
        measured_bytes=256,
        producer=producer,
    )
    evidence = make_qualification_evidence(
        export=reference_for_record(export),
        runtime_contract=runtime,
        device_profile=qualification_device or device,
        workload=workload,
        placement=placement,
        check_id="numerical_parity",
        outcome=check_outcome,
        measurement={"receipt": _reference("parity-receipt")},
        producer=producer,
    )
    qualification = make_device_qualification(
        export=reference_for_record(export),
        target_profile=reference_for_record(target),
        runtime_contract=runtime,
        device_profile=qualification_device or device,
        workload=workload,
        placement=placement,
        required_checks=["numerical_parity"],
        checks=[
            {
                "id": "numerical_parity",
                "outcome": check_outcome,
                "evidence": [reference_for_record(evidence)],
            }
        ],
        producer=producer,
    )
    market = make_market_snapshot(
        observed_at="2026-08-24T00:00:00Z",
        sources=[
            {
                "source_id": "steam-hardware-survey",
                "scope": "respondent_share",
                "collection_method": "published aggregate survey",
                "raw_receipt": _reference("steam-raw"),
                "observations": {"rtx_5060_share": 0.01},
            }
        ],
        producer=producer,
    )
    release = make_release_decision(
        collection_contract=reference_for_record(contract),
        market_snapshot=reference_for_record(market),
        policy=_reference("release-policy"),
        included=[
            {
                "variant_key": "rtx5060-8gb",
                "alias": "qwen3.8-27b-gridbook-8gb",
                "target_profile": reference_for_record(target),
                "export": reference_for_record(export),
                "qualification": reference_for_record(qualification),
            }
        ],
        rejected=[],
        producer=producer,
    )
    by_name = {
        "ledger": ledger,
        "model": model,
        "probe": probe,
        "candidate": candidate,
        "catalog": catalog,
        "cost": cost,
        "target": target,
        "contract": contract,
        "solve": solve,
        "export": export,
        "evidence": evidence,
        "qualification": qualification,
        "market": market,
        "release": release,
    }
    return list(by_name.values()), by_name


def test_complete_collection_graph_reconciles_and_cli_verifies(tmp_path, capsys):
    records, _ = _collection_records()
    result = verify_collection_graph(records)
    assert result["record_count"] == 14
    assert result["identity_sha256"] == verify_collection_graph(
        list(reversed(records))
    )["identity_sha256"]

    paths = []
    for index, record in enumerate(records):
        path = tmp_path / f"{index:02d}.json"
        write_record(path, record)
        paths.append(str(path))
    assert collection_cli(["verify", *paths]) == 0
    assert '"record_count": 14' in capsys.readouterr().out


def test_graph_refuses_missing_or_duplicate_owned_records():
    records, by_name = _collection_records()
    without_candidate = [record for record in records if record is not by_name["candidate"]]
    with pytest.raises(ArtifactCollectionError, match="is absent"):
        verify_collection_graph(without_candidate)
    with pytest.raises(ArtifactCollectionError, match="more than once"):
        verify_collection_graph([*records, by_name["candidate"]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"probe_features": ("h_trace_raw",)}, "probe is insufficient"),
        ({"omit_cost_cell": True}, "cost matrix is not exhaustive"),
        ({"include_assignment": False}, "every required unit exactly once"),
        ({"cost_status": "unavailable"}, "no usable cost cell"),
        ({"assignment_local_bytes": 0}, "local bytes differ"),
        ({"assignment_resources": ()}, "shared resources differ"),
        (
            {"qualification_device": _reference("different-gpu")},
            "device/workload differs from target",
        ),
        ({"check_outcome": "failed"}, "included artifact is not qualified"),
    ],
)
def test_graph_refuses_cross_record_drift(kwargs, message):
    records, _ = _collection_records(**kwargs)
    with pytest.raises(ArtifactCollectionError, match=message):
        verify_collection_graph(records)


def test_export_refuses_incomplete_whole_artifact_byte_measurement():
    _, records = _collection_records()
    with pytest.raises(ArtifactCollectionError, match="complete file inventory"):
        make_export_record(
            solve=records["solve"],
            target_profile=reference_for_record(records["target"]),
            artifact=_reference("bad-export"),
            files=[
                {
                    "path": "model.safetensors",
                    "size_bytes": 256,
                    "sha256": _digest("model.safetensors"),
                }
            ],
            tensors=[],
            codebooks=[],
            runtime_artifact=_reference("runtime-artifact"),
            byte_scope="recursive_package_bytes",
            measured_bytes=255,
            producer=_reference("producer"),
        )


def test_hugging_face_incidence_cannot_be_labeled_installed_base():
    with pytest.raises(ArtifactCollectionError, match="not an installed-base"):
        make_market_snapshot(
            observed_at="2026-08-24T00:00:00Z",
            sources=[
                {
                    "source_id": "huggingface-user-agent-survey",
                    "scope": "installed-base estimate",
                    "collection_method": "self-reported library telemetry",
                    "raw_receipt": _reference("hf-raw"),
                    "observations": {"rtx_5060": 10},
                }
            ],
            producer=_reference("producer"),
        )


def test_unknown_record_schema_is_not_an_unvalidated_escape_hatch():
    with pytest.raises(ArtifactCollectionError, match="unsupported"):
        seal_record("fixture.unknown.v1", {"accepted": True})


def test_zero_check_qualification_cannot_be_accepted():
    with pytest.raises(ArtifactCollectionError, match="must not be empty"):
        make_device_qualification(
            export=make_reference(
                subject_schema=EXPORT_SCHEMA,
                subject_id=_digest("empty-check-export"),
                content_sha256=_digest("empty-check-export-content"),
                size_bytes=1,
            ),
            target_profile=_reference("target"),
            runtime_contract=_reference("runtime"),
            device_profile=_reference("device"),
            workload=_reference("workload"),
            placement={"mode": "single_gpu"},
            required_checks=[],
            checks=[],
            producer=_reference("producer"),
        )


def test_loaded_export_cannot_point_a_tensor_outside_file_inventory():
    _, records = _collection_records()
    payload = copy.deepcopy(records["export"]["payload"])
    payload["tensors"][0]["file"] = "missing.safetensors"
    with pytest.raises(ArtifactCollectionError, match="outside the inventory"):
        seal_record(EXPORT_SCHEMA, payload)


@pytest.mark.parametrize("path", ["/model.safetensors", "../model.bin", "a/../b"])
def test_export_inventory_paths_are_portable_and_relative(path):
    _, records = _collection_records()
    with pytest.raises(ArtifactCollectionError, match="portable relative path"):
        make_export_record(
            solve=records["solve"],
            target_profile=reference_for_record(records["target"]),
            artifact=_reference("bad-path-export"),
            files=[
                {
                    "path": path,
                    "size_bytes": 1,
                    "sha256": _digest("bad-path-file"),
                }
            ],
            tensors=[],
            codebooks=[],
            runtime_artifact=_reference("runtime"),
            byte_scope="recursive_package_bytes",
            measured_bytes=1,
            producer=_reference("producer"),
        )


def test_contract_graph_refuses_lineage_drift_even_when_envelope_is_valid():
    records, by_name = _collection_records()
    other_probe = make_probe_campaign(
        model_snapshot=reference_for_record(by_name["model"]),
        probe_blob=_reference("other-probe"),
        calibration=_reference("other-calibration"),
        token_content=_reference("other-tokens"),
        measured_features=["act_sq_sum"],
        covered_unit_ids=["body.0.linear"],
        missing_unit_ids=[],
        merge_receipt=_reference("other-merge"),
        producer=_reference("producer"),
    )
    payload = copy.deepcopy(by_name["contract"]["payload"])
    payload["probe_campaign"] = reference_for_record(other_probe)
    drifted_contract = seal_record(CONTRACT_SCHEMA, payload)
    graph = [
        record
        for record in records
        if record is not by_name["contract"] and record is not by_name["release"]
    ]
    with pytest.raises(ArtifactCollectionError, match="cost binds a different"):
        verify_collection_graph([*graph, other_probe, drifted_contract])


def test_receipt_and_manifest_graphs_reconcile_exact_contract_variants():
    records, by_name = _collection_records()
    unknown_variant_receipt = make_stage_receipt(
        collection_contract=reference_for_record(by_name["contract"]),
        variant_key="not-a-variant",
        stage="export",
        outcome="accepted",
        inputs=[reference_for_record(by_name["solve"])],
        outputs=[reference_for_record(by_name["export"])],
        evidence=[_reference("export-evidence")],
        producer=_reference("producer"),
    )
    with pytest.raises(ArtifactCollectionError, match="unknown collection variant"):
        verify_collection_graph([*records, unknown_variant_receipt])

    other_contract = make_collection_contract(
        model_snapshot=reference_for_record(by_name["model"]),
        probe_campaign=reference_for_record(by_name["probe"]),
        candidate_catalog=reference_for_record(by_name["catalog"]),
        cost_snapshot=reference_for_record(by_name["cost"]),
        accounting_rule=by_name["target"]["payload"]["accounting_rule"],
        variants={
            "other": {
                "target_profile": by_name["target"],
                "export_contract": _reference("other-export-contract"),
            }
        },
    )
    receipt = make_stage_receipt(
        collection_contract=reference_for_record(other_contract),
        variant_key="other",
        stage="export",
        outcome="accepted",
        inputs=[reference_for_record(by_name["solve"])],
        outputs=[reference_for_record(by_name["export"])],
        evidence=[_reference("other-export-evidence")],
        producer=_reference("producer"),
    )
    valid_other_manifest = make_collection_manifest(
        collection_contract=other_contract,
        receipts=[receipt],
    )
    manifest_payload = copy.deepcopy(valid_other_manifest["payload"])
    manifest_payload["collection_contract"] = reference_for_record(
        by_name["contract"]
    )
    drifted_manifest = seal_record(MANIFEST_SCHEMA, manifest_payload)
    with pytest.raises(ArtifactCollectionError, match="different collection contract"):
        verify_collection_graph(
            [*records, other_contract, receipt, drifted_manifest]
        )


def test_qualification_evidence_cannot_be_replayed_across_runtime():
    records, by_name = _collection_records()
    replayed_evidence = make_qualification_evidence(
        export=reference_for_record(by_name["export"]),
        runtime_contract=_reference("other-runtime"),
        device_profile=by_name["target"]["payload"]["device_profile"],
        workload=by_name["target"]["payload"]["workload"],
        placement=by_name["target"]["payload"]["placement_constraints"],
        check_id="numerical_parity",
        outcome="passed",
        measurement={"receipt": _reference("replayed-receipt")},
        producer=_reference("producer"),
    )
    replayed_qualification = make_device_qualification(
        export=reference_for_record(by_name["export"]),
        target_profile=reference_for_record(by_name["target"]),
        runtime_contract=by_name["export"]["payload"]["runtime_artifact"],
        device_profile=by_name["target"]["payload"]["device_profile"],
        workload=by_name["target"]["payload"]["workload"],
        placement=by_name["target"]["payload"]["placement_constraints"],
        required_checks=["numerical_parity"],
        checks=[
            {
                "id": "numerical_parity",
                "outcome": "passed",
                "evidence": [reference_for_record(replayed_evidence)],
            }
        ],
        producer=_reference("producer"),
    )
    with pytest.raises(ArtifactCollectionError, match="evidence bindings differ"):
        verify_collection_graph(
            [*records, replayed_evidence, replayed_qualification]
        )


def test_release_decision_must_exhaust_its_bound_contract():
    records, by_name = _collection_records()
    invalid_release = make_release_decision(
        collection_contract=reference_for_record(by_name["contract"]),
        market_snapshot=reference_for_record(by_name["market"]),
        policy=_reference("release-policy"),
        included=[],
        rejected=[
            {
                "variant_key": "invented",
                "target_profile": reference_for_record(by_name["target"]),
                "reason": "not in the collection",
            }
        ],
        producer=_reference("producer"),
    )
    without_release = [
        record for record in records if record is not by_name["release"]
    ]
    with pytest.raises(ArtifactCollectionError, match="differs from contract"):
        verify_collection_graph([*without_release, invalid_release])


def test_exhaustive_source_and_catalog_records_cannot_be_empty():
    with pytest.raises(ArtifactCollectionError, match="must not be empty"):
        make_unit_ledger(model_content_sha256=_digest("model"), units=[])
    with pytest.raises(ArtifactCollectionError, match="must not be empty"):
        make_candidate_catalog([])


def test_graph_refuses_cross_record_physical_size_equivocation():
    content_digest = _digest("same-physical-content")
    first_reference = make_reference(
        subject_schema="fixture.blob.v1",
        subject_id=_digest("first-subject"),
        content_sha256=content_digest,
        size_bytes=10,
    )
    second_reference = make_reference(
        subject_schema="fixture.blob.v1",
        subject_id=_digest("second-subject"),
        content_sha256=content_digest,
        size_bytes=11,
    )
    first = make_auxiliary("first", {"object": first_reference})
    second = make_auxiliary("second", {"object": second_reference})
    with pytest.raises(ArtifactCollectionError, match="inconsistent sizes"):
        verify_collection_graph([first, second])


def test_market_snapshot_requires_canonical_utc_timestamp():
    with pytest.raises(ArtifactCollectionError, match="canonical UTC"):
        make_market_snapshot(
            observed_at="2026-08-24",
            sources=[
                {
                    "source_id": "steam",
                    "scope": "respondent_share",
                    "collection_method": "published aggregate survey",
                    "raw_receipt": _reference("steam"),
                    "observations": {"rtx_5060_share": 0.01},
                }
            ],
            producer=_reference("producer"),
        )
