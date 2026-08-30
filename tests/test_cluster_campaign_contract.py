from __future__ import annotations

from copy import deepcopy
import json

import pytest

from prismaquant.cluster_campaign_contract import (
    CAMPAIGN_MANIFEST_SCHEMA,
    CAMPAIGN_STATE_SCHEMA,
    STAGE_DAG,
    ClusterCampaignContractError,
    StageAssignment,
    canonical_sha256,
    complete_assignment,
    initial_campaign_state,
    next_ready_assignments,
    parse_campaign_manifest,
    seal_campaign_manifest,
    validate_campaign_manifest,
    validate_campaign_state,
)


def _host(
    host_id: str,
    *,
    local: bool,
    suffix: str,
) -> dict[str, object]:
    producer_commit = "1" * 40
    image_digest = "sha256:" + "4" * 64
    transport: dict[str, object]
    if local:
        transport = {"kind": "local"}
    else:
        transport = {
            "kind": "ssh",
            "host": "peer.example.test",
            "port": 22,
            "user": "campaign_runner",
        }
    return {
        "id": host_id,
        "transport": transport,
        "roots": {
            "model_root": f"/srv/prismaquant-{suffix}/model",
            "dataset_path": f"/srv/prismaquant-{suffix}/data/calibration.pt",
            "snapshot_root": f"/srv/prismaquant-{suffix}/snapshot",
            "run_root": f"/srv/prismaquant-{suffix}/run",
            "worker_state_root": f"/srv/prismaquant-{suffix}/worker-state",
        },
        "expected": {
            "hostname": f"rtx4090-{suffix}",
            "gpu": {
                "name": "NVIDIA GeForce RTX 4090",
                "uuid": f"GPU-{suffix.upper()}-0123456789",
                "compute_capability": [8, 9],
                "device_count": 1,
            },
            "image_digest": image_digest,
            "producer_commit": producer_commit,
            "uid": 1000,
            "gid": 1000,
        },
    }


def _manifest_body(*, reverse_hosts: bool = False) -> dict[str, object]:
    hosts = [
        _host("alpha", local=True, suffix="alpha"),
        _host("zeta", local=False, suffix="zeta"),
    ]
    if reverse_hosts:
        hosts.reverse()
    return {
        "schema": CAMPAIGN_MANIFEST_SCHEMA,
        "campaign_id": "qwen38-27b-fp8-burn-001",
        "coordinator": "alpha",
        "producer": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "snapshot_sha256": "3" * 64,
            "image_digest": "sha256:" + "4" * 64,
        },
        "inputs": {
            "model_content_sha256": "5" * 64,
            "dataset_sha256": "6" * 64,
            "sample_parallel": {
                "nsamples": 128,
                "seqlen": 2048,
                "calib_seed": 42,
                "activation_rows_limit": 1024,
            },
        },
        "hosts": hosts,
    }


def _receipt(assignment: StageAssignment) -> str:
    return canonical_sha256(
        {"work_id": assignment.work_id, "result": "complete"}
    )


def _resign_state(state: dict[str, object]) -> dict[str, object]:
    body = {key: value for key, value in state.items() if key != "identity_sha256"}
    return {**body, "identity_sha256": canonical_sha256(body)}


def _finish_frontier(
    manifest: dict[str, object],
    state: dict[str, object],
) -> dict[str, object]:
    for assignment in next_ready_assignments(manifest, state):
        state = complete_assignment(
            manifest, state, assignment, _receipt(assignment)
        )
    return state


def test_manifest_is_strictly_validated_and_host_order_is_canonical() -> None:
    manifest = seal_campaign_manifest(_manifest_body())
    reversed_manifest = seal_campaign_manifest(
        _manifest_body(reverse_hosts=True)
    )

    assert manifest == reversed_manifest
    assert [host["id"] for host in manifest["hosts"]] == ["alpha", "zeta"]
    assert validate_campaign_manifest(manifest) == manifest
    assert parse_campaign_manifest(json.dumps(manifest)) == manifest

    with_extra = {**manifest, "unexpected": True}
    with pytest.raises(ClusterCampaignContractError, match="fields differ"):
        validate_campaign_manifest(with_extra)

    duplicate_json = json.dumps(manifest).replace(
        '"schema":', '"schema": "duplicate", "schema":', 1
    )
    with pytest.raises(ClusterCampaignContractError, match="duplicate JSON member"):
        parse_campaign_manifest(duplicate_json)


def test_manifest_rejects_duplicate_hosts_and_identity_aliases() -> None:
    duplicate_ids = _manifest_body()
    duplicate_ids["hosts"][1]["id"] = "alpha"
    with pytest.raises(ClusterCampaignContractError, match="duplicate ids"):
        seal_campaign_manifest(duplicate_ids)

    duplicate_hostnames = _manifest_body()
    duplicate_hostnames["hosts"][1]["expected"]["hostname"] = "rtx4090-alpha"
    with pytest.raises(
        ClusterCampaignContractError, match="duplicate expected hostnames"
    ):
        seal_campaign_manifest(duplicate_hostnames)

    duplicate_gpus = _manifest_body()
    duplicate_gpus["hosts"][1]["expected"]["gpu"]["uuid"] = (
        "GPU-ALPHA-0123456789"
    )
    with pytest.raises(ClusterCampaignContractError, match="duplicate expected GPU"):
        seal_campaign_manifest(duplicate_gpus)


@pytest.mark.parametrize(
    "unsafe",
    [
        "relative/run",
        "/",
        "/srv/prismaquant/../run",
        "/srv/prismaquant/run/",
        "/srv//prismaquant/run",
        "/srv/prismaquant/run;unsafe",
    ],
)
def test_manifest_rejects_unsafe_absolute_roots(unsafe: str) -> None:
    body = _manifest_body()
    body["hosts"][0]["roots"]["run_root"] = unsafe
    with pytest.raises(ClusterCampaignContractError, match="absolute|normalized"):
        seal_campaign_manifest(body)


def test_manifest_rejects_overlapping_writable_roots() -> None:
    body = _manifest_body()
    body["hosts"][0]["roots"]["run_root"] = "/srv/prismaquant-alpha/work"
    body["hosts"][0]["roots"]["worker_state_root"] = (
        "/srv/prismaquant-alpha/work/state"
    )
    with pytest.raises(ClusterCampaignContractError, match="overlaps"):
        seal_campaign_manifest(body)


def test_fixed_dag_traversal_completes_every_assignment_once() -> None:
    manifest = seal_campaign_manifest(_manifest_body(reverse_hosts=True))
    state = initial_campaign_state(manifest)
    observed_stages: list[str] = []

    while ready := next_ready_assignments(manifest, state):
        assert len({item.stage for item in ready}) == 1
        observed_stages.append(ready[0].stage)
        for assignment in ready:
            state = complete_assignment(
                manifest, state, assignment, _receipt(assignment)
            )

    assert observed_stages == [item.stage for item in STAGE_DAG]
    assert len(state["completions"]) == 21
    assert next_ready_assignments(manifest, state) == ()
    assert validate_campaign_state(state, manifest) == state


def test_parallel_stage_is_a_barrier_and_ready_order_is_deterministic() -> None:
    manifest = seal_campaign_manifest(_manifest_body(reverse_hosts=True))
    state = initial_campaign_state(manifest)
    ready = next_ready_assignments(manifest, state)

    assert [(item.stage, item.host_id) for item in ready] == [
        ("host_preflight", "alpha"),
        ("host_preflight", "zeta"),
    ]
    state = complete_assignment(manifest, state, ready[1], _receipt(ready[1]))
    assert next_ready_assignments(manifest, state) == (ready[0],)

    state = complete_assignment(manifest, state, ready[0], _receipt(ready[0]))
    next_stage = next_ready_assignments(manifest, state)
    assert [(item.stage, item.host_id) for item in next_stage] == [
        ("prepare_calibration", "alpha")
    ]


def test_parallel_completion_order_produces_identical_state() -> None:
    manifest = seal_campaign_manifest(_manifest_body())
    initial = initial_campaign_state(manifest)
    first, second = next_ready_assignments(manifest, initial)

    state_forward = complete_assignment(
        manifest, initial, first, _receipt(first)
    )
    state_forward = complete_assignment(
        manifest, state_forward, second, _receipt(second)
    )
    state_reverse = complete_assignment(
        manifest, initial, second, _receipt(second)
    )
    state_reverse = complete_assignment(
        manifest, state_reverse, first, _receipt(first)
    )

    assert state_forward == state_reverse
    assert state_forward["identity_sha256"] == state_reverse["identity_sha256"]


@pytest.mark.parametrize("drift", ["gpu", "root", "commit"])
def test_state_rejects_manifest_identity_drift(drift: str) -> None:
    manifest = seal_campaign_manifest(_manifest_body())
    state = initial_campaign_state(manifest)
    assignment = next_ready_assignments(manifest, state)[0]
    state = complete_assignment(manifest, state, assignment, _receipt(assignment))
    changed = _manifest_body()
    if drift == "gpu":
        changed["hosts"][0]["expected"]["gpu"]["uuid"] = "GPU-DRIFT-0001"
    elif drift == "root":
        changed["hosts"][0]["roots"]["run_root"] = "/srv/drift/run"
    else:
        changed["producer"]["commit"] = "a" * 40
        for host in changed["hosts"]:
            host["expected"]["producer_commit"] = "a" * 40
    drifted_manifest = seal_campaign_manifest(changed)

    with pytest.raises(ClusterCampaignContractError, match="different campaign"):
        validate_campaign_state(state, drifted_manifest)


def test_state_rejects_duplicate_completion_and_identity_tampering() -> None:
    manifest = seal_campaign_manifest(_manifest_body())
    state = initial_campaign_state(manifest)
    assignment = next_ready_assignments(manifest, state)[0]
    state = complete_assignment(manifest, state, assignment, _receipt(assignment))

    duplicate = deepcopy(state)
    duplicate["completions"].append(deepcopy(duplicate["completions"][0]))
    duplicate = _resign_state(duplicate)
    with pytest.raises(ClusterCampaignContractError, match="duplicate work_id"):
        validate_campaign_state(duplicate, manifest)

    tampered = deepcopy(state)
    tampered["completions"][0]["host_identity_sha256"] = "f" * 64
    tampered = _resign_state(tampered)
    with pytest.raises(ClusterCampaignContractError, match="manifest host"):
        validate_campaign_state(tampered, manifest)


def test_state_rejects_dependency_skip_even_with_valid_state_hash() -> None:
    manifest = seal_campaign_manifest(_manifest_body())
    initial = initial_campaign_state(manifest)
    after_preflight = _finish_frontier(manifest, initial)
    prepare = next_ready_assignments(manifest, after_preflight)[0]
    after_prepare = complete_assignment(
        manifest, after_preflight, prepare, _receipt(prepare)
    )
    prepare_completion = deepcopy(after_prepare["completions"][-1])
    skipped = {
        "schema": CAMPAIGN_STATE_SCHEMA,
        "campaign_identity_sha256": manifest["identity_sha256"],
        "completions": [prepare_completion],
    }
    skipped = _resign_state(skipped)

    with pytest.raises(ClusterCampaignContractError, match="violates barrier"):
        validate_campaign_state(skipped, manifest)

    with pytest.raises(ClusterCampaignContractError, match="current.*frontier"):
        complete_assignment(manifest, initial, prepare, _receipt(prepare))
