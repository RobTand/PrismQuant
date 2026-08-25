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
    bind_gridbook_runtime_contract,
    canonical_sha256,
    seal_campaign_manifest,
)
from prismaquant.cluster_host_admission import (
    ClusterHostAdmissionError,
    GPU_START_GUARD_RECEIPT_SCHEMA,
    HOST_ADMISSION_RECEIPT_SCHEMA,
    HOST_PRE_ADMISSION_RECEIPT_SCHEMA,
    MODEL_IDENTITY_RECEIPT_SCHEMA,
    HostAdmissionRuntime,
    acquire_gpu_lease,
    admit_host,
    adopt_gpu_lease,
    build_host_action_request,
    guard_gpu_start,
    guarded_container_launch,
    inspect_gpu_lease,
    pre_admit_host,
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
        "artifact_target": {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "compute_capability": [8, 9],
            "artifact_max_bytes": 18_000_000_000,
            "disposition": "validation_only",
            "source_dtype": "bf16",
            "physical_formats": [
                "FP8_CB_K4", "FP8_CB_K16", "FP8_CB_K48", "FP8_E4M3",
            ],
            "terminal_format": "BF16",
            "allocation_objective": "context_first",
        },
        "producer": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "snapshot_sha256": "3" * 64,
            "image_digest": "sha256:" + "4" * 64,
        },
        "inputs": {
            "model_content_sha256": model_sha256,
            "dataset_sha256": dataset_sha256,
            "gridbook_runtime_contract": bind_gridbook_runtime_contract({
                "schema": "gridbook.runtime-contract.v11",
                "test_fixture": True,
            }),
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
    def __init__(
        self,
        *,
        compute_apps: bytes = b"",
        gpu_rows: bytes | None = None,
        campaign_containers: bytes = b"",
    ):
        self.compute_apps = compute_apps
        self.campaign_containers = campaign_containers
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
        if argv[:2] == ("docker", "ps"):
            return _Completed(stdout=self.campaign_containers)
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
    repeated = release_gpu_lease(
        first, "alpha", lease_root=lease_root,
    )
    assert repeated["disposition"] == "already_absent"
    assert repeated["lease"] == released["lease"] == acquired["lease"]
    with pytest.raises(
        ClusterHostAdmissionError, match="already released.*reacquisition",
    ):
        acquire_gpu_lease(first, "alpha", lease_root=lease_root)

    never_held = release_gpu_lease(
        first, "zeta", lease_root=lease_root,
    )
    assert never_held["disposition"] == "already_absent"
    assert never_held["lease"] is None


def test_release_retry_proves_post_unlink_receipt_crash_with_exact_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prismaquant.cluster_host_admission as admission

    lease_root = tmp_path / "leases"
    manifest = _manifest(tmp_path)
    acquired = acquire_gpu_lease(manifest, "alpha", lease_root=lease_root)
    original_fsync_directory = admission._fsync_directory
    fsync_calls = 0

    def fail_after_unlink(path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("injected post-unlink receipt crash")
        original_fsync_directory(path)

    monkeypatch.setattr(admission, "_fsync_directory", fail_after_unlink)
    with pytest.raises(ClusterHostAdmissionError, match="cannot release"):
        release_gpu_lease(manifest, "alpha", lease_root=lease_root)

    monkeypatch.setattr(admission, "_fsync_directory", original_fsync_directory)
    recovered = release_gpu_lease(
        manifest, "alpha", lease_root=lease_root,
    )

    assert recovered["disposition"] == "already_absent"
    assert recovered["lease"] == acquired["lease"]


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


def test_pre_admission_defers_model_cache_but_guard_rechecks_idle_gpu(
    tmp_path: Path,
) -> None:
    dataset = b'{"text":"calibration"}\n'
    _prepare_alpha_files(tmp_path, dataset)
    cache = tmp_path / "alpha" / "worker-state" / "source-identity-cache.json"
    cache.unlink()
    manifest = _manifest(
        tmp_path, dataset_sha256=hashlib.sha256(dataset).hexdigest(),
    )
    lease_root = tmp_path / "leases"
    runner = _NvidiaRunner()

    admitted = pre_admit_host(
        manifest,
        "alpha",
        lease_root=lease_root,
        runtime=_runtime(runner=runner),
    )
    assert admitted["schema"] == HOST_PRE_ADMISSION_RECEIPT_SCHEMA
    assert admitted["lease"]["disposition"] == "acquired"
    assert not cache.exists()

    guarded = guard_gpu_start(
        manifest,
        "alpha",
        lease_root=lease_root,
        runtime=_runtime(runner=runner),
    )
    assert guarded["schema"] == GPU_START_GUARD_RECEIPT_SCHEMA
    assert guarded["lease"]["disposition"] == "adopted"

    busy = _NvidiaRunner(
        compute_apps=b"GPU-ALPHA-0123456789, 4242, python3\n",
    )
    with pytest.raises(ClusterHostAdmissionError, match="compute-apps is nonempty"):
        guard_gpu_start(
            manifest,
            "alpha",
            lease_root=lease_root,
            runtime=_runtime(runner=busy),
        )


def test_guard_refuses_a_labeled_orphan_even_when_gpu_is_still_idle(
    tmp_path: Path,
) -> None:
    dataset = b'{"text":"calibration"}\n'
    _prepare_alpha_files(tmp_path, dataset)
    manifest = _manifest(
        tmp_path, dataset_sha256=hashlib.sha256(dataset).hexdigest(),
    )
    lease_root = tmp_path / "leases"
    acquire_gpu_lease(manifest, "alpha", lease_root=lease_root)
    runner = _NvidiaRunner(campaign_containers=b"0123456789ab\n")

    with pytest.raises(ClusterHostAdmissionError, match="still running"):
        guard_gpu_start(
            manifest,
            "alpha",
            lease_root=lease_root,
            runtime=_runtime(runner=runner),
        )


def test_guarded_launch_carries_the_lease_lock_into_fixed_foreground_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prismaquant.cluster_host_admission as admission

    manifest = _manifest(tmp_path)
    lease_root = tmp_path / "leases"
    acquire_gpu_lease(manifest, "alpha", lease_root=lease_root)
    runner = _NvidiaRunner()
    campaign_label = f"io.prismaquant.campaign={manifest['identity_sha256']}"
    command = (
        "docker", "run", "--rm",
        "--label", campaign_label,
        "--label", "io.prismaquant.host=alpha",
        "--label", "io.prismaquant.work=sample_ce:alpha",
        "eugr/spark-vllm@sha256:" + "4" * 64,
    )
    observed = {}

    monkeypatch.setattr(
        admission.os,
        "set_inheritable",
        lambda descriptor, value: observed.update(
            descriptor=descriptor, inheritable=value,
        ),
    )

    def control(argv, **kwargs):
        observed.setdefault("control", []).append((tuple(argv), dict(kwargs)))
        if argv[:2] == ["docker", "inspect"]:
            return _Completed(returncode=1, stderr=b"No such object")
        if argv[:3] == ["docker", "ps", "-a"]:
            return _Completed(stdout=b"")
        raise AssertionError(argv)

    class Process:
        pid = 4242

        def __init__(self, argv, **kwargs):
            observed.update(argv=tuple(argv), kwargs=dict(kwargs))
            cid_path = Path(argv[argv.index("--cidfile") + 1])
            cid_path.write_text("a" * 64 + "\n", encoding="ascii")

        def wait(self, *, timeout):
            observed["wait_timeout"] = timeout
            return 0

    def popen(argv, **kwargs):
        return Process(argv, **kwargs)

    result = guarded_container_launch(
        manifest,
        "alpha",
        command,
        work_id="sample_ce:alpha",
        timeout_seconds=86400.0,
        lease_root=lease_root,
        runtime=_runtime(runner=runner),
        popen_impl=popen,
        run_impl=control,
    )

    assert result == 0
    assert observed["descriptor"] >= 0
    assert observed["inheritable"] is True
    assert observed["argv"][:3] == command[:3]
    assert observed["argv"][3] == "--cidfile"
    assert observed["argv"][5] == "--name"
    assert observed["argv"][7:] == command[3:]
    assert observed["kwargs"] == {
        "shell": False,
        "close_fds": True,
        "pass_fds": (observed["descriptor"],),
        "start_new_session": True,
    }
    assert observed["wait_timeout"] == 86400.0
    assert not Path(observed["argv"][4]).exists()
    assert [call[0] for call in runner.calls] == [
        "nvidia-smi", "nvidia-smi", "docker",
    ]


def test_guarded_launch_timeout_stops_exact_cid_and_retires_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prismaquant.cluster_host_admission as admission

    manifest = _manifest(tmp_path)
    lease_root = tmp_path / "leases"
    acquire_gpu_lease(manifest, "alpha", lease_root=lease_root)
    work_id = "measure_burn:alpha"
    cid = "b" * 64
    command = (
        "docker", "run", "--rm",
        "--label", f"io.prismaquant.campaign={manifest['identity_sha256']}",
        "--label", "io.prismaquant.host=alpha",
        "--label", f"io.prismaquant.work={work_id}",
        "eugr/spark-vllm@sha256:" + "4" * 64,
    )
    state = {"active": False, "control": [], "signals": []}

    def inspect_payload():
        return json.dumps([{
            "Config": {"Labels": {
                "io.prismaquant.campaign": manifest["identity_sha256"],
                "io.prismaquant.host": "alpha",
                "io.prismaquant.work": work_id,
            }},
            "State": {"Running": True},
        }]).encode()

    def control(argv, **kwargs):
        del kwargs
        argv = tuple(argv)
        state["control"].append(argv)
        if argv[:2] == ("docker", "inspect"):
            return (
                _Completed(stdout=inspect_payload())
                if state["active"] else _Completed(returncode=1)
            )
        if argv[:3] == ("docker", "ps", "-a"):
            return _Completed(stdout=b"")
        if argv[:2] == ("docker", "stop"):
            assert argv[-1] == cid
            state["active"] = False
            return _Completed()
        raise AssertionError(argv)

    class Process:
        pid = 5252

        def __init__(self, argv):
            cid_path = Path(argv[argv.index("--cidfile") + 1])
            cid_path.write_text(cid + "\n", encoding="ascii")
            state["active"] = True
            self.waits = 0

        def wait(self, *, timeout):
            self.waits += 1
            if self.waits == 1:
                raise admission.subprocess.TimeoutExpired("docker", timeout)
            return -15

    monkeypatch.setattr(
        admission.os,
        "killpg",
        lambda pid, sig: state["signals"].append((pid, sig)),
    )

    result = guarded_container_launch(
        manifest,
        "alpha",
        command,
        work_id=work_id,
        timeout_seconds=1.0,
        lease_root=lease_root,
        runtime=_runtime(runner=_NvidiaRunner()),
        popen_impl=lambda argv, **_kwargs: Process(argv),
        run_impl=control,
    )

    assert result == 124
    assert any(call[:2] == ("docker", "stop") for call in state["control"])
    assert state["signals"]
    assert not list(lease_root.rglob("*.cid"))


def test_guarded_launch_reconciles_exact_owned_orphan_before_idle_gate(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    lease_root = tmp_path / "leases"
    acquire_gpu_lease(manifest, "alpha", lease_root=lease_root)
    work_id = "measure_burn:alpha"
    stale_cid = "c" * 64
    work_key = hashlib.sha256(work_id.encode("utf-8")).hexdigest()
    supervision = (
        lease_root / "container-supervision" / manifest["identity_sha256"]
        / "alpha"
    )
    supervision.mkdir(parents=True, mode=0o700)
    (supervision / f"{work_key}.cid").write_text(
        stale_cid + "\n", encoding="ascii",
    )
    command = (
        "docker", "run", "--rm",
        "--label", f"io.prismaquant.campaign={manifest['identity_sha256']}",
        "--label", "io.prismaquant.host=alpha",
        "--label", f"io.prismaquant.work={work_id}",
        "eugr/spark-vllm@sha256:" + "4" * 64,
    )
    state = {"active": True, "idle_checks": [], "stopped": []}

    class DynamicRunner(_NvidiaRunner):
        def __call__(self, argv: tuple[str, ...]) -> _Completed:
            if any(item.startswith("--query-compute-apps=") for item in argv):
                state["idle_checks"].append(state["active"])
                return _Completed(
                    stdout=(
                        b"GPU-ALPHA-0123456789, 4242, python3\n"
                        if state["active"] else b""
                    )
                )
            if argv[:2] == ("docker", "ps"):
                return _Completed(
                    stdout=(stale_cid[:12] + "\n").encode()
                    if state["active"] else _Completed().stdout
                )
            return super().__call__(argv)

    def inspect_payload():
        return json.dumps([{
            "Config": {"Labels": {
                "io.prismaquant.campaign": manifest["identity_sha256"],
                "io.prismaquant.host": "alpha",
                "io.prismaquant.work": work_id,
            }},
            "State": {"Running": True},
        }]).encode()

    def control(argv, **_kwargs):
        argv = tuple(argv)
        if argv[:2] == ("docker", "inspect"):
            return (
                _Completed(stdout=inspect_payload())
                if state["active"] else _Completed(returncode=1)
            )
        if argv[:3] == ("docker", "ps", "-a"):
            return _Completed(stdout=b"")
        if argv[:2] == ("docker", "stop"):
            state["stopped"].append(argv[-1])
            state["active"] = False
            return _Completed()
        raise AssertionError(argv)

    class Process:
        pid = 6262

        def __init__(self, argv):
            Path(argv[argv.index("--cidfile") + 1]).write_text(
                "d" * 64 + "\n", encoding="ascii",
            )

        def wait(self, *, timeout):
            assert timeout == 10.0
            return 0

    result = guarded_container_launch(
        manifest,
        "alpha",
        command,
        work_id=work_id,
        timeout_seconds=10.0,
        lease_root=lease_root,
        runtime=_runtime(runner=DynamicRunner()),
        popen_impl=lambda argv, **_kwargs: Process(argv),
        run_impl=control,
    )

    assert result == 0
    assert state["stopped"] == [stale_cid]
    assert state["idle_checks"] == [False]


def test_guarded_launch_requires_ownership_labels_before_the_image(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    lease_root = tmp_path / "leases"
    acquire_gpu_lease(manifest, "alpha", lease_root=lease_root)
    work_id = "sample_ce:alpha"
    command = (
        "docker", "run", "--rm",
        "--label", "foreign.a=1",
        "--label", "foreign.b=2",
        "--label", "foreign.c=3",
        "eugr/spark-vllm@sha256:" + "4" * 64,
        f"io.prismaquant.campaign={manifest['identity_sha256']}",
        "io.prismaquant.host=alpha",
        f"io.prismaquant.work={work_id}",
    )

    with pytest.raises(
        ClusterHostAdmissionError, match="fixed foreground container shape",
    ):
        guarded_container_launch(
            manifest,
            "alpha",
            command,
            work_id=work_id,
            timeout_seconds=10.0,
            lease_root=lease_root,
            runtime=_runtime(runner=_NvidiaRunner()),
            popen_impl=lambda *_args, **_kwargs: pytest.fail(
                "malformed Docker command launched"
            ),
        )


def test_guarded_launch_rejects_label_alias_override_before_image(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    lease_root = tmp_path / "leases"
    acquire_gpu_lease(manifest, "alpha", lease_root=lease_root)
    work_id = "sample_ce:alpha"
    command = (
        "docker", "run", "--rm",
        "--label", f"io.prismaquant.campaign={manifest['identity_sha256']}",
        "--label", "io.prismaquant.host=alpha",
        "--label", f"io.prismaquant.work={work_id}",
        "-l", "io.prismaquant.campaign=foreign",
        "eugr/spark-vllm@sha256:" + "4" * 64,
    )

    with pytest.raises(
        ClusterHostAdmissionError, match="fixed foreground container shape",
    ):
        guarded_container_launch(
            manifest,
            "alpha",
            command,
            work_id=work_id,
            timeout_seconds=10.0,
            lease_root=lease_root,
            runtime=_runtime(runner=_NvidiaRunner()),
            popen_impl=lambda *_args, **_kwargs: pytest.fail(
                "label-alias Docker command launched"
            ),
        )


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

    def validate(
        model, cache, *, require_complete_checkpoint, cached_source_model,
    ):
        calls.append((
            model, cache, require_complete_checkpoint, cached_source_model,
        ))
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

    assert calls == [(
        tmp_path / "model", tmp_path / "cache.json", True, "/model",
    )]
    assert result == _model_receipt()


@pytest.mark.parametrize(
    "action", ["pre-admit", "admit", "guard", "inspect", "adopt", "release"],
)
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
    assert request.timeout_seconds == 300.0
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
