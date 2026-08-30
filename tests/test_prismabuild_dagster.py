from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from prismaquant import prismabuild as pb
from prismaquant import prismabuild_dagster as pd
from prismaquant import prismabuild_slurm as ps


def _action(
    checkout: Path,
    name: str,
    *,
    inputs: list[dict[str, object]] | None = None,
    portability: str = "portable",
    platform_key: str | None = None,
    host_class: str | None = None,
) -> dict[str, object]:
    code = checkout / f"{name}.py"
    code.write_text(f"# {name} closure\n", encoding="utf-8")
    argv = [
        sys.executable,
        "-c",
        f"open('{name}.bin','wb').write(b'{name}')",
    ]
    toolchain: dict[str, str] = {}
    if portability != "portable":
        toolchain.update(pb.executable_toolchain_contract(argv[0]))
        evidence = pb._collect_worker_evidence()
        toolchain.update(
            {
                "system": str(evidence["system"]),
                "machine": str(evidence["machine"]),
                "libc": str(evidence["libc"]),
            }
        )
        accelerators = evidence["accelerators"]
        if accelerators:
            toolchain.update(
                {
                    "cuda_compute_capability": accelerators[0]["compute_capability"],
                    "nvidia_driver": accelerators[0]["driver_version"],
                }
            )
    return pb.seal_action(
        {
            "schema": pb.ACTION_SCHEMA_V1,
            "task": {
                "definition_id": f"tests/dagster/{name}",
                "definition_version": "v1",
                "task_class": "generation",
                "determinism": "deterministic",
                "artifact_kind": "generic",
                "argv": argv,
                "working_directory": ".",
                "result_path": f"{name}.bin",
            },
            "inputs": inputs or [],
            "code_closure": pb.build_code_closure(checkout, [f"{name}.py"]),
            "params": {"name": name},
            "environment": {
                "variables": {"PATH": "/usr/bin:/bin"},
                "toolchain": toolchain,
            },
            "execution_scope": {
                "portability": portability,
                "platform_key": platform_key,
                "host_class": host_class,
            },
        }
    )


def _resources(host_class: str = "cpu") -> ps.SlurmResources:
    return ps.SlurmResources(
        cpus=2,
        memory_mib=4096,
        gpus=0,
        constraint=host_class,
        partition=host_class,
        account="prismaquant",
        qos="batch",
        time_limit="00:10:00",
    )


def _spec(
    action: object,
    checkout: Path,
    *,
    dependencies: tuple[pd.CASDependency, ...] = (),
    max_requeues: int = 0,
) -> pd.ActionSpec:
    return pd.ActionSpec(
        action=action,
        checkout_root=checkout,
        resources=_resources(),
        placement=ps.SlurmPlacement(
            platform_key=None, host_class=None
        ),
        dependencies=dependencies,
        max_requeues=max_requeues,
        poll_interval_seconds=0.001,
        max_polls=3,
    )


def _publish(
    cas_root: Path,
    spec: pd.ActionSpec,
    source: Path,
    payload: bytes,
) -> dict[str, object]:
    source.write_bytes(payload)
    attestation = pb.preflight_action(
        spec.action,
        cas_root=cas_root,
        checkout_root=spec.checkout_root,
    )
    receipt, _ = pb.PrismaBuildCAS(cas_root).publish_result(
        spec.action,
        source,
        attestation=attestation,
    )
    return receipt


class _NeverAdapter:
    def submit(self, *args: object, **kwargs: object) -> ps.SlurmSubmission:
        raise AssertionError("cache hit must not call SLURM")

    def resolve(self, *args: object, **kwargs: object) -> ps.SlurmResolution:
        raise AssertionError("cache hit must not call SLURM")

    def requeue(self, *args: object, **kwargs: object) -> bool:
        raise AssertionError("cache hit must not call SLURM")

    def cancel(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("cache hit must not call SLURM")


def test_import_has_no_dagster_dependency_or_import_time_side_effect():
    code = (
        "import sys; import prismaquant.prismabuild_dagster; "
        "assert 'dagster' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_native_definitions_fail_clearly_when_dagster_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    action = _action(tmp_path, "root")
    graph = pd.ActionGraph([_spec(action, tmp_path)])
    real_import = pd.importlib.import_module

    def missing(name: str, *args: object, **kwargs: object) -> Any:
        if name == "dagster":
            raise ModuleNotFoundError("dagster")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(pd.importlib, "import_module", missing)
    with pytest.raises(pd.DagsterUnavailableError, match="optional"):
        pd.build_dagster_definitions(graph)


@pytest.mark.skipif(
    importlib.util.find_spec("dagster") is None,
    reason="optional Dagster package is not installed",
)
def test_native_definitions_materialize_only_from_verified_cache(tmp_path: Path):
    spec = _spec(_action(tmp_path, "root"), tmp_path)
    graph = pd.ActionGraph([spec], name="native_cache_smoke")
    cas_root = tmp_path / "cas"
    _publish(cas_root, spec, tmp_path / "source", b"cached")
    definitions = pd.build_dagster_definitions(graph)
    assert [asset.key.path for asset in definitions.assets] == [
        ["prismabuild", spec.action_key]
    ]
    job = definitions.resolve_job_def("native_cache_smoke_job")
    result = job.execute_in_process(
        run_config={
            "resources": {
                "prismabuild": {
                    "config": {
                        "cas_root": str(cas_root),
                        "log_root": str(tmp_path / "logs"),
                        "worker_script": str(tmp_path / "missing-worker-is-unused"),
                    }
                }
            }
        }
    )
    assert result.success


def test_graph_order_is_deterministic_and_edges_are_content_bound(tmp_path: Path):
    digest = hashlib.sha256(b"root-result").hexdigest()
    root = _spec(_action(tmp_path, "root"), tmp_path)
    independent = _spec(_action(tmp_path, "independent"), tmp_path)
    dependency = pd.CASDependency(
        upstream_action_key=root.action_key,
        input_id="prismabuild/root",
        result_sha256=digest,
        result_bytes=len(b"root-result"),
    )
    child = _spec(
        _action(
            tmp_path,
            "child",
            inputs=[
                {
                    "id": dependency.input_id,
                    "sha256": dependency.result_sha256,
                    "bytes": dependency.result_bytes,
                }
            ],
        ),
        tmp_path,
        dependencies=(dependency,),
    )
    graph_a = pd.ActionGraph([child, root, independent], name="campaign")
    graph_b = pd.ActionGraph([independent, child, root], name="campaign")
    keys_a = [node.action_key for node in pd.definition_plan(graph_a)]
    keys_b = [node.action_key for node in pd.definition_plan(graph_b)]
    assert keys_a == keys_b
    assert keys_a.index(root.action_key) < keys_a.index(child.action_key)
    child_plan = next(
        node for node in pd.definition_plan(graph_a) if node.action_key == child.action_key
    )
    assert child_plan.dependency_keys == (root.action_key,)
    assert all("/" not in component for component in child_plan.asset_path[1:])


def test_graph_refuses_dependency_not_exactly_bound_in_action_inputs(tmp_path: Path):
    root = _spec(_action(tmp_path, "root"), tmp_path)
    child = _spec(
        _action(tmp_path, "child"),
        tmp_path,
        dependencies=(
            pd.CASDependency(
                upstream_action_key=root.action_key,
                input_id="prismabuild/root",
                result_sha256="a" * 64,
                result_bytes=1,
            ),
        ),
    )
    with pytest.raises(pd.DagsterGraphError, match="action.inputs"):
        pd.ActionGraph([root, child])


def test_action_spec_config_round_trip_binds_placement_and_retries(tmp_path: Path):
    action = _action(tmp_path, "root")
    spec = _spec(action, tmp_path, max_requeues=2)
    rebuilt = pd.ActionSpec.from_config(action, spec.as_config())
    assert rebuilt.action_key == spec.action_key
    assert rebuilt.as_config() == spec.as_config()
    malformed = spec.as_config()
    malformed["placement"] = {
        "platform_key": None,
        "host_class": None,
        "mutable_path": "/tmp/result",
    }
    with pytest.raises(pd.DagsterGraphError, match="fields differ"):
        pd.ActionSpec.from_config(action, malformed)


def test_verified_cache_hit_short_circuits_without_adapter(tmp_path: Path):
    action = _action(tmp_path, "root")
    spec = _spec(action, tmp_path)
    graph = pd.ActionGraph([spec])
    cas_root = tmp_path / "cas"
    receipt = _publish(cas_root, spec, tmp_path / "source", b"cached")
    runner = pd.DagsterActionRunner(cas_root=cas_root, adapter=_NeverAdapter())
    result = runner.execute(graph, spec.action_key)
    assert result.action_key == spec.action_key
    assert result.result_sha256 == receipt["result"]["sha256"]  # type: ignore[index]


class _FalseSuccessAdapter:
    def __init__(self, action_key: str):
        self.action_key = action_key
        self.job = ps.SlurmJobId(41)

    def submit(self, *args: object, **kwargs: object) -> ps.SlurmSubmission:
        return ps.SlurmSubmission("submitted", self.action_key, self.job, None)

    def resolve(self, *args: object, **kwargs: object) -> ps.SlurmResolution:
        return ps.SlurmResolution(
            "succeeded",
            self.action_key,
            self.job,
            "COMPLETED",
            "orchestrator says success",
            None,
            None,
        )

    def requeue(self, *args: object, **kwargs: object) -> bool:
        raise AssertionError("success path does not requeue")


def test_orchestrator_success_without_receipt_fails_closed(tmp_path: Path):
    spec = _spec(_action(tmp_path, "root"), tmp_path)
    runner = pd.DagsterActionRunner(
        cas_root=tmp_path / "cas",
        adapter=_FalseSuccessAdapter(spec.action_key),
    )
    with pytest.raises(pd.DagsterActionError, match="without a verified CAS receipt"):
        runner.execute(pd.ActionGraph([spec]), spec.action_key)


class _RequeueAdapter:
    def __init__(self, *, cas_root: Path, spec: pd.ActionSpec, source: Path):
        self.cas_root = cas_root
        self.spec = spec
        self.source = source
        self.job = ps.SlurmJobId(73)
        self.action_keys: list[str] = []
        self.job_ids: list[ps.SlurmJobId | str] = []
        self.requeued = False

    def submit(self, action: object, **kwargs: object) -> ps.SlurmSubmission:
        normalized = pb.validate_action(action)
        self.action_keys.append(str(normalized["action_key"]))
        return ps.SlurmSubmission("submitted", self.spec.action_key, self.job, None)

    def resolve(
        self, action: object, job_id: ps.SlurmJobId | str
    ) -> ps.SlurmResolution:
        normalized = pb.validate_action(action)
        self.action_keys.append(str(normalized["action_key"]))
        self.job_ids.append(job_id)
        if not self.requeued:
            return ps.SlurmResolution(
                "failed",
                self.spec.action_key,
                self.job,
                "NODE_FAIL",
                "test failure",
                None,
                None,
            )
        receipt = pb.PrismaBuildCAS(self.cas_root).lookup(self.spec.action)
        assert receipt is not None
        return ps.SlurmResolution(
            "succeeded",
            self.spec.action_key,
            self.job,
            None,
            "verified CAS receipt exists",
            receipt,
            pb.PrismaBuildCAS(self.cas_root).result_path(receipt, self.spec.action),
        )

    def requeue(self, action: object, job_id: ps.SlurmJobId | str) -> bool:
        normalized = pb.validate_action(action)
        self.action_keys.append(str(normalized["action_key"]))
        self.job_ids.append(job_id)
        _publish(self.cas_root, self.spec, self.source, b"after-requeue")
        self.requeued = True
        return True

    def cancel(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("successful retry must not cancel")


def test_retry_requeues_same_job_and_preserves_exact_action_identity(tmp_path: Path):
    spec = _spec(_action(tmp_path, "root"), tmp_path, max_requeues=1)
    cas_root = tmp_path / "cas"
    adapter = _RequeueAdapter(
        cas_root=cas_root, spec=spec, source=tmp_path / "retry-source"
    )
    runner = pd.DagsterActionRunner(
        cas_root=cas_root, adapter=adapter, sleep=lambda _: None
    )
    result = runner.execute(pd.ActionGraph([spec]), spec.action_key)
    assert result.action_key == spec.action_key
    assert adapter.action_keys == [spec.action_key] * len(adapter.action_keys)
    assert adapter.job_ids == [adapter.job, adapter.job, adapter.job]


class _LaggingRequeueAdapter:
    def __init__(self, action_key: str):
        self.action_key = action_key
        self.job = ps.SlurmJobId(74)
        self.states = iter(["failed", "failed", "pending", "failed"])
        self.requeues = 0
        self.cancelled = False

    def submit(self, *args: object, **kwargs: object) -> ps.SlurmSubmission:
        return ps.SlurmSubmission("submitted", self.action_key, self.job, None)

    def resolve(self, *args: object, **kwargs: object) -> ps.SlurmResolution:
        state = next(self.states)
        return ps.SlurmResolution(
            state, self.action_key, self.job, "NODE_FAIL", "test state", None, None
        )

    def requeue(self, *args: object, **kwargs: object) -> bool:
        self.requeues += 1
        return True

    def cancel(self, *args: object, **kwargs: object) -> None:
        self.cancelled = True


def test_requeue_waits_for_scheduler_transition_before_terminal_retry(tmp_path: Path):
    spec = _spec(_action(tmp_path, "root"), tmp_path, max_requeues=1)
    adapter = _LaggingRequeueAdapter(spec.action_key)
    sleeps: list[float] = []
    runner = pd.DagsterActionRunner(
        cas_root=tmp_path / "cas", adapter=adapter, sleep=sleeps.append
    )
    with pytest.raises(pd.DagsterActionError, match="ended failed"):
        runner.execute(pd.ActionGraph([spec]), spec.action_key)
    assert adapter.requeues == 1
    assert adapter.cancelled is False
    assert len(sleeps) == 3


class _PollingAdapter:
    def __init__(self, action_key: str):
        self.action_key = action_key
        self.job = ps.SlurmJobId(75)
        self.cancelled: list[ps.SlurmJobId | str] = []

    def submit(self, *args: object, **kwargs: object) -> ps.SlurmSubmission:
        return ps.SlurmSubmission("submitted", self.action_key, self.job, None)

    def resolve(self, *args: object, **kwargs: object) -> ps.SlurmResolution:
        return ps.SlurmResolution(
            "pending", self.action_key, self.job, "NOT_VISIBLE",
            "accounting lag", None, None,
        )

    def requeue(self, *args: object, **kwargs: object) -> bool:
        raise AssertionError("pending allocation must not requeue")

    def cancel(self, job_id: ps.SlurmJobId | str) -> None:
        self.cancelled.append(job_id)


def test_poll_budget_cancels_exact_allocation_before_failing(tmp_path: Path):
    spec = _spec(_action(tmp_path, "root"), tmp_path)
    adapter = _PollingAdapter(spec.action_key)
    runner = pd.DagsterActionRunner(
        cas_root=tmp_path / "cas", adapter=adapter, sleep=lambda _: None
    )
    with pytest.raises(pd.DagsterActionError, match="allocation was cancelled"):
        runner.execute(pd.ActionGraph([spec]), spec.action_key)
    assert adapter.cancelled == [adapter.job]


def test_missing_upstream_receipt_fails_before_downstream_submission(tmp_path: Path):
    digest = "b" * 64
    root = _spec(_action(tmp_path, "root"), tmp_path)
    dependency = pd.CASDependency(
        upstream_action_key=root.action_key,
        input_id="prismabuild/root",
        result_sha256=digest,
        result_bytes=7,
    )
    child = _spec(
        _action(
            tmp_path,
            "child",
            inputs=[{"id": dependency.input_id, "sha256": digest, "bytes": 7}],
        ),
        tmp_path,
        dependencies=(dependency,),
    )
    runner = pd.DagsterActionRunner(
        cas_root=tmp_path / "cas", adapter=_NeverAdapter()
    )
    with pytest.raises(pd.DagsterActionError, match="has no CAS receipt"):
        runner.execute(pd.ActionGraph([root, child]), child.action_key)


def test_tampered_cached_payload_fails_closed_before_adapter(tmp_path: Path):
    spec = _spec(_action(tmp_path, "root"), tmp_path)
    cas_root = tmp_path / "cas"
    receipt = _publish(cas_root, spec, tmp_path / "source", b"canonical")
    digest = receipt["result"]["sha256"]  # type: ignore[index]
    blob = cas_root / "blobs" / str(digest)[:2] / str(digest)
    blob.chmod(0o644)
    blob.write_bytes(b"tampered")
    runner = pd.DagsterActionRunner(cas_root=cas_root, adapter=_NeverAdapter())
    with pytest.raises(pb.CASTamperError, match="payload content differs"):
        runner.execute(pd.ActionGraph([spec]), spec.action_key)


def test_wrong_scope_receipt_fails_closed_before_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    action = _action(
        tmp_path, "root", portability="host_class_keyed", host_class="gb10"
    )
    spec = pd.ActionSpec(
        action=action,
        checkout_root=tmp_path,
        resources=_resources("gb10"),
        placement=ps.SlurmPlacement(
            platform_key="linux-aarch64-sm121", host_class="gb10"
        ),
    )
    cas_root = tmp_path / "cas"
    source = tmp_path / "source"
    source.write_bytes(b"canonical")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "sparky")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gb10")
    monkeypatch.setattr(
        pb,
        "_verify_slurm_process_membership",
        lambda job_id: f"/slurm/job_{job_id}/step_batch",
    )
    attestation = pb.preflight_action(
        spec.action, cas_root=cas_root, checkout_root=spec.checkout_root
    )
    cas = pb.PrismaBuildCAS(cas_root)
    receipt, _ = cas.publish_result(
        spec.action,
        source,
        attestation=attestation,
    )
    altered = dict(receipt)
    producer = json.loads(json.dumps(receipt["producer"]))
    producer["host_class"] = "rtx4090"
    producer["evidence"]["slurm"]["partition"] = "rtx4090"
    producer_body = {
        key: producer[key] for key in producer if key != "attestation_sha256"
    }
    producer["attestation_sha256"] = hashlib.sha256(
        json.dumps(
            producer_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    altered["producer"] = producer
    body = {key: altered[key] for key in altered if key != "receipt_sha256"}
    altered["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    receipt_path = cas._receipt_path(spec.action_key)
    receipt_path.chmod(0o644)
    receipt_path.write_text(
        json.dumps(altered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    runner = pd.DagsterActionRunner(cas_root=cas_root, adapter=_NeverAdapter())
    with pytest.raises(pb.CASTamperError, match="host_class"):
        runner.execute(pd.ActionGraph([spec]), spec.action_key)
