from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant.cluster_campaign_contract import (
    CAMPAIGN_MANIFEST_SCHEMA,
    canonical_sha256,
    seal_campaign_manifest,
)
from prismaquant.cluster_host_admission import (
    ClusterHostAdmissionError,
    HOST_ADMISSION_RECEIPT_SCHEMA,
    MODEL_IDENTITY_RECEIPT_SCHEMA,
    HostAdmissionRuntime,
    acquire_gpu_lease,
    admit_host,
    adopt_gpu_lease,
    build_host_action_request,
    inspect_gpu_lease,
    release_gpu_lease,
)
from prismaquant.cluster_transport import RunRequest


GIB = 1024**3
MODEL_SHA = "a" * 64
GPU_UUID = "GPU-ALPHA-0123456789"


def _host(host_id: str, root: Path, *, local: bool) -> dict[str, object]:
    transport: dict[str, object] = (
        {"kind": "local"}
        if local
        else {
            "kind": "ssh",
            "host": "peer.example.test",
            "port": 22,
            "user": "campaign_runner",
        }
    )
    gpu_uuid = GPU_UUID if host_id == "alpha" else "GPU-ZETA-0123456789"
    gpu_name = "NVIDIA GeForce RTX 4090" if host_id == "alpha" else "NVIDIA GB10"
    capability = [8, 9] if host_id == "alpha" else [12, 1]
    return {
        "id": host_id,
        "transport": transport,
        "roots": {
            "model_root": str(root / "model"),
            "dataset_path": str(root / "data" / "calibration.jsonl"),
            "snapshot_root": str(root / "snapshot"),
            "run_root": str(root / "run"),
            "worker_state_root": str(root / "worker-state"),
        },
        "expected": {
            "hostname": f"{host_id}-host",
            "gpu": {
                "name": gpu_name,
                "uuid": gpu_uuid,
                "compute_capability": capability,
                "device_count": 1,
            },
            "image_digest": "sha256:" + "4" * 64,
            "producer_commit": "1" * 40,
            "uid": 1000,
            "gid": 1000,
        },
    }


def _body(
    tmp_path: Path,
    *,
    campaign_id: str = "qwen38-host-admission",
    dataset_sha256: str = "b" * 64,
    model_sha256: str = MODEL_SHA,
) -> dict[str, object]:
    return {
        "schema": CAMPAIGN_MANIFEST_SCHEMA,
        "campaign_id": campaign_id,
        "coordinator": "alpha",
        "producer": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "snapshot_sha256": "3" * 64,
            "image_digest": "sha256:" + "4" * 64,
        },
        "inputs": {
            "model_content_sha256": model_sha256,
            "dataset_sha256": dataset_sha256,
            "sample_parallel": {
                "nsamples": 32,
                "seqlen": 1024,
                "calib_seed": 42,
                "activation_rows_limit": 1024,
            },
        },
        "policy": {
            "retry": {"max_attempts": 2},
            "telemetry": {
                "interval_milliseconds": 1000,
                "require_positive_gpu_utilization": True,
            },
            "resources": {
                "coordinator_min_free_bytes": 100 * GIB,
                "worker_min_free_bytes": 40 * GIB,
                "min_mem_available_bytes": 64 * GIB,
            },
            "outputs": {
                "owner": "coordinator",
                "transfer_mode": "sha256_no_clobber",
            },
        },
        "hosts": [
            _host("zeta", tmp_path / "zeta", local=False),
            _host("alpha", tmp_path / "alpha", local=True),
        ],
    }


def _manifest(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    return seal_campaign_manifest(_body(tmp_path, **kwargs))


def _prepare_alpha_files(tmp_path: Path, dataset: bytes) -> None:
    root = tmp_path / "alpha"
    for name in ("model", "snapshot", "run", "worker-state", "data"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "data" / "calibration.jsonl").write_bytes(dataset)
    (root / "worker-state" / "source-identity-cache.json").write_text(
        '{"fixture":"trusted-by-injected-reader"}\n', encoding="utf-8",
    )


def _model_receipt(portable_sha256: str = MODEL_SHA) -> dict[str, object]:
    return {
        "schema": MODEL_IDENTITY_RECEIPT_SCHEMA,
        "identity": {
            "schema": "prismaquant.streamed_model.identity.v1",
            "content_sha256": "c" * 64,
            "resolved_commit": None,
            "checkpoint_shards": 16,
            "checkpoint_tensors": 100,
        },
        "portable": {
            "schema": "prismaquant.streamed_model.portable_content.v1",
            "portable_content_sha256": portable_sha256,
            "checkpoint_shards": 16,
            "checkpoint_tensors": 100,
        },
    }


@dataclass
class _Completed:
    returncode: int = 0
    stdout: bytes = b""
    stderr: bytes = b""


class _NvidiaRunner:
    def __init__(self, *, compute_apps: bytes = b"", gpu_rows: bytes | None = None):
        self.compute_apps = compute_apps
        self.gpu_rows = gpu_rows or (
            b"0, NVIDIA GeForce RTX 4090, GPU-ALPHA-0123456789, 8.9\n"
        )
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> _Completed:
        self.calls.append(argv)
        if any(item.startswith("--query-gpu=") for item in argv):
            return _Completed(stdout=self.gpu_rows)
        if any(item.startswith("--query-compute-apps=") for item in argv):
            return _Completed(stdout=self.compute_apps)
        raise AssertionError(f"unexpected fixed command: {argv}")


def _runtime(
    *,
    runner: _NvidiaRunner | None = None,
    free_bytes: int = 200 * GIB,
    mem_available: int = 128 * GIB,
    model_sha256: str = MODEL_SHA,
) -> HostAdmissionRuntime:
    return HostAdmissionRuntime(
        command_runner=runner or _NvidiaRunner(),
        disk_usage_reader=lambda _path: SimpleNamespace(free=free_bytes),
        meminfo_reader=lambda: f"MemAvailable: {mem_available // 1024} kB\n",
        hostname_reader=lambda: "alpha-host",
        uid_reader=lambda: 1000,
        gid_reader=lambda: 1000,
        model_identity_reader=lambda _model, _cache: _model_receipt(model_sha256),
    )


def test_durable_gpu_lease_is_adoptable_and_excludes_competing_campaign(
    tmp_path: Path,
) -> None:
    lease_root = tmp_path / "leases"
    first = _manifest(tmp_path)
    competing = _manifest(tmp_path, campaign_id="competing-campaign")

    acquired = acquire_gpu_lease(first, "alpha", lease_root=lease_root)
    assert acquired["disposition"] == "acquired"
    assert inspect_gpu_lease(GPU_UUID, lease_root=lease_root)["lease"] == acquired["lease"]
    assert acquire_gpu_lease(
        first, "alpha", lease_root=lease_root,
    )["disposition"] == "adopted"
    assert adopt_gpu_lease(
        first, "alpha", lease_root=lease_root,
    )["lease"] == acquired["lease"]

    with pytest.raises(ClusterHostAdmissionError, match="different sealed"):
        acquire_gpu_lease(competing, "alpha", lease_root=lease_root)
    with pytest.raises(ClusterHostAdmissionError, match="another sealed"):
        release_gpu_lease(competing, "alpha", lease_root=lease_root)

    released = release_gpu_lease(first, "alpha", lease_root=lease_root)
    assert released["disposition"] == "released"
    assert inspect_gpu_lease(GPU_UUID, lease_root=lease_root)["disposition"] == "absent"
    assert release_gpu_lease(
        first, "alpha", lease_root=lease_root,
    )["disposition"] == "already_absent"


def test_corrupt_or_noncanonical_lease_fails_closed(tmp_path: Path) -> None:
    lease_root = tmp_path / "leases"
    manifest = _manifest(tmp_path)
    acquire_gpu_lease(manifest, "alpha", lease_root=lease_root)
    path = next(lease_root.glob("*.json"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    with pytest.raises(ClusterHostAdmissionError, match="canonically encoded"):
        inspect_gpu_lease(GPU_UUID, lease_root=lease_root)


def test_host_admission_binds_gpu_resources_dataset_model_and_lease(
    tmp_path: Path,
) -> None:
    dataset = b'{"text":"calibration"}\n'
    _prepare_alpha_files(tmp_path, dataset)
    manifest = _manifest(
        tmp_path, dataset_sha256=hashlib.sha256(dataset).hexdigest(),
    )
    runner = _NvidiaRunner()

    receipt = admit_host(
        manifest,
        "alpha",
        lease_root=tmp_path / "leases",
        runtime=_runtime(runner=runner),
    )

    assert receipt["schema"] == HOST_ADMISSION_RECEIPT_SCHEMA
    assert receipt["identity_sha256"] == canonical_sha256({
        key: value for key, value in receipt.items() if key != "identity_sha256"
    })
    assert receipt["gpu"] == {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "uuid": GPU_UUID,
        "compute_capability": "8.9",
        "device_count": 1,
    }
    assert receipt["compute_apps"]["rows"] == 0
    assert receipt["resources"]["role"] == "coordinator"
    assert {
        row["required_free_bytes"] for row in receipt["resources"]["disks"]
    } == {100 * GIB}
    assert receipt["dataset"]["sha256"] == hashlib.sha256(dataset).hexdigest()
    assert receipt["model"]["portable"]["portable_content_sha256"] == MODEL_SHA
    assert receipt["lease"]["disposition"] == "acquired"
    assert [call[1].split("=")[0] for call in runner.calls] == [
        "--query-gpu", "--query-compute-apps",
    ]


def test_nonempty_compute_apps_refuses_before_lease(tmp_path: Path) -> None:
    dataset = b"calibration\n"
    _prepare_alpha_files(tmp_path, dataset)
    manifest = _manifest(
        tmp_path, dataset_sha256=hashlib.sha256(dataset).hexdigest(),
    )
    runner = _NvidiaRunner(
        compute_apps=b"GPU-ALPHA-0123456789, 4242, python3\n",
    )

    with pytest.raises(ClusterHostAdmissionError, match="compute-apps is nonempty"):
        admit_host(
            manifest, "alpha", lease_root=tmp_path / "leases",
            runtime=_runtime(runner=runner),
        )
    assert inspect_gpu_lease(
        GPU_UUID, lease_root=tmp_path / "leases",
    )["disposition"] == "absent"


def test_live_gpu_count_and_identity_are_exact(tmp_path: Path) -> None:
    dataset = b"calibration\n"
    _prepare_alpha_files(tmp_path, dataset)
    manifest = _manifest(
        tmp_path, dataset_sha256=hashlib.sha256(dataset).hexdigest(),
    )
    two_rows = (
        b"0, NVIDIA GeForce RTX 4090, GPU-ALPHA-0123456789, 8.9\n"
        b"1, NVIDIA GeForce RTX 4090, GPU-OTHER-0123456789, 8.9\n"
    )

    with pytest.raises(ClusterHostAdmissionError, match="exactly one visible"):
        admit_host(
            manifest, "alpha", lease_root=tmp_path / "leases",
            runtime=_runtime(runner=_NvidiaRunner(gpu_rows=two_rows)),
        )


@pytest.mark.parametrize(
    ("free_bytes", "mem_available", "match"),
    [
        (99 * GIB, 128 * GIB, "free bytes"),
        (200 * GIB, 63 * GIB, "MemAvailable"),
    ],
)
def test_resource_minima_fail_closed_before_lease(
    tmp_path: Path,
    free_bytes: int,
    mem_available: int,
    match: str,
) -> None:
    dataset = b"calibration\n"
    _prepare_alpha_files(tmp_path, dataset)
    manifest = _manifest(
        tmp_path, dataset_sha256=hashlib.sha256(dataset).hexdigest(),
    )

    with pytest.raises(ClusterHostAdmissionError, match=match):
        admit_host(
            manifest, "alpha", lease_root=tmp_path / "leases",
            runtime=_runtime(
                free_bytes=free_bytes, mem_available=mem_available,
            ),
        )


@pytest.mark.parametrize("which", ["dataset", "model"])
def test_content_identity_mismatch_fails_before_lease(
    tmp_path: Path, which: str,
) -> None:
    dataset = b"calibration\n"
    _prepare_alpha_files(tmp_path, dataset)
    dataset_sha = hashlib.sha256(dataset).hexdigest()
    manifest = _manifest(
        tmp_path,
        dataset_sha256="d" * 64 if which == "dataset" else dataset_sha,
    )
    runtime = _runtime(model_sha256="e" * 64 if which == "model" else MODEL_SHA)

    with pytest.raises(ClusterHostAdmissionError, match="dataset|streamed-model"):
        admit_host(
            manifest, "alpha", lease_root=tmp_path / "leases", runtime=runtime,
        )


def test_trusted_model_reader_reuses_complete_streamed_identity_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prismaquant.cluster_host_admission as admission
    import prismaquant.cost_streaming as streaming

    calls: list[tuple[object, ...]] = []
    identity = {"schema": "prismaquant.streamed_model.identity.v1"}

    def validate(model, cache, *, require_complete_checkpoint):
        calls.append((model, cache, require_complete_checkpoint))
        return identity

    monkeypatch.setattr(streaming, "validate_cached_streamed_model_identity", validate)
    monkeypatch.setattr(
        streaming,
        "compact_streamed_model_identity",
        lambda value, *, where: _model_receipt()["identity"],
    )
    monkeypatch.setattr(
        streaming,
        "portable_streamed_model_content_identity",
        lambda value, *, where: _model_receipt()["portable"],
    )
    result = admission._trusted_model_identity(
        tmp_path / "model", tmp_path / "cache.json",
    )

    assert calls == [(tmp_path / "model", tmp_path / "cache.json", True)]
    assert result == _model_receipt()


@pytest.mark.parametrize("action", ["admit", "inspect", "adopt", "release"])
def test_host_action_request_is_fixed_safe_and_transport_roundtrippable(
    tmp_path: Path, action: str,
) -> None:
    manifest = _manifest(tmp_path)
    request = build_host_action_request(action, manifest, "alpha", operation_index=7)

    assert request.argv[:4] == ("python3", "-P", "-B", "-s")
    assert request.argv[-4:] == (
        "prismaquant.cluster_host_admission", action, "--host-id", "alpha",
    )
    assert request.argv[4] == (
        f"{tmp_path}/alpha/snapshot/tools/prismaquant_source_bootstrap.py"
    )
    assert request.cwd == "/"
    assert request.inherit_env is False
    assert dict(request.env)["PYTHONSAFEPATH"] == "1"
    assert json.loads(request.stdin) == manifest
    assert RunRequest.from_payload(request.to_payload()) == request
    assert request.job_id.endswith(f"-alpha-007")


def test_action_builder_rejects_arbitrary_action_or_attempt(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ClusterHostAdmissionError, match="unsupported"):
        build_host_action_request("shell", manifest, "alpha")  # type: ignore[arg-type]
    with pytest.raises(ClusterHostAdmissionError, match="operation_index"):
        build_host_action_request("admit", manifest, "alpha", operation_index=-1)
