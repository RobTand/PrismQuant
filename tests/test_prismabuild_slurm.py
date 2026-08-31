from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from prismaquant import prismabuild as pb
from prismaquant import prismabuild_slurm as ps


def _action(
    checkout: Path,
    *,
    portability: str = "portable",
    platform_key: str | None = None,
    host_class: str | None = None,
) -> dict[str, object]:
    code = checkout / "task.py"
    code.write_text("# immutable closure\n", encoding="utf-8")
    argv = [sys.executable, "-c", "open('result.bin','wb').write(b'ok')"]
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
                "definition_id": "tests/slurm",
                "definition_version": "v1",
                "task_class": "generation",
                "determinism": "deterministic",
                "artifact_kind": "generic",
                "argv": argv,
                "working_directory": ".",
                "result_path": "result.bin",
            },
            "inputs": [],
            "code_closure": pb.build_code_closure(checkout, ["task.py"]),
            "params": {},
            "environment": {
                "variables": {"DECLARED": "1"},
                "toolchain": toolchain,
            },
            "execution_scope": {
                "portability": portability,
                "platform_key": platform_key,
                "host_class": host_class,
            },
        }
    )


def _attestation(
    checkout: Path, action: dict[str, object], cas_root: Path
) -> dict[str, object]:
    return pb.preflight_action(
        action, cas_root=cas_root, checkout_root=checkout
    )


def _resources(**changes: Any) -> ps.SlurmResources:
    values: dict[str, object] = {
        "cpus": 8,
        "memory_mib": 65536,
        "gpus": 1,
        "constraint": "gb10",
        "partition": "gb10",
        "account": "prismaquant",
        "qos": "gold",
        "time_limit": "04:00:00",
    }
    values.update(changes)
    return ps.SlurmResources(**values)  # type: ignore[arg-type]


def _adapter(tmp_path: Path) -> tuple[ps.SlurmAdapter, Path]:
    worker = tmp_path / "prismabuild-worker"
    worker.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    worker.chmod(0o555)
    return (
        ps.SlurmAdapter(
            cas_root=tmp_path / "cas",
            log_root=tmp_path / "logs",
            worker_script=worker,
            cluster="gold-cluster",
            sbatch="/slurm/bin/sbatch",
            squeue="/slurm/bin/squeue",
            sacct="/slurm/bin/sacct",
            scancel="/slurm/bin/scancel",
            scontrol="/slurm/bin/scontrol",
        ),
        worker,
    )


def _restart_adapter(adapter: ps.SlurmAdapter) -> ps.SlurmAdapter:
    return ps.SlurmAdapter(
        cas_root=adapter.cas_root,
        log_root=adapter.log_root,
        worker_script=adapter.worker_script,
        cluster=adapter.cluster,
        sbatch=adapter.sbatch,
        squeue=adapter.squeue,
        sacct=adapter.sacct,
        scancel=adapter.scancel,
        scontrol=adapter.scontrol,
        submit_environment=adapter.submit_environment,
        command_timeout_seconds=adapter.command_timeout_seconds,
    )


def _record_intent(
    adapter: ps.SlurmAdapter,
    action: dict[str, object],
    checkout: Path,
    *,
    max_polls: int = 1,
    max_requeues: int = 0,
    poll_interval_seconds: float = 5.0,
) -> dict[str, object]:
    """Install the pre-sbatch append-only state for an action."""

    key = str(action["action_key"])
    request = ps.publish_action_request(action, cas_root=adapter.cas_root)
    scope = action["execution_scope"]
    assert isinstance(scope, dict)
    placement = ps.SlurmPlacement(
        platform_key=scope["platform_key"],  # type: ignore[arg-type]
        host_class=scope["host_class"],  # type: ignore[arg-type]
    )
    submit_spec = adapter._submit_spec(
        action_key=key,
        request_path=request,
        checkout_root=checkout,
        resources=_resources(),
        placement=placement,
        max_polls=max_polls,
        max_requeues=max_requeues,
        poll_interval_seconds=poll_interval_seconds,
        recompute=False,
    )
    intent = adapter._submission_intent(submit_spec)
    adapter._publish_submission_intent(intent)
    return intent


def _record_submission(
    adapter: ps.SlurmAdapter,
    action: dict[str, object],
    checkout: Path,
    job_id: str,
    *,
    max_polls: int = 1,
    max_requeues: int = 0,
    poll_interval_seconds: float = 5.0,
) -> dict[str, object]:
    """Install the state that a successful submit would publish."""

    intent = _record_intent(
        adapter,
        action,
        checkout,
        max_polls=max_polls,
        max_requeues=max_requeues,
        poll_interval_seconds=poll_interval_seconds,
    )
    parsed = ps.parse_job_id(job_id)
    if parsed.cluster is None:
        parsed = ps.SlurmJobId(parsed.number, adapter.cluster)
    adapter._publish_job_binding(intent, parsed)
    return intent


def _state_row(
    intent: dict[str, object], job_id: int, state: str
) -> str:
    return f"{job_id}|{state}|{intent['job_name']}|{intent['comment']}\n"


def _completed(
    argv: list[str], stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=stderr)


def test_submit_uses_exact_argv_closed_environment_and_content_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    action = _action(
        checkout,
        portability="host_class_keyed",
        host_class="gb10",
    )
    adapter, worker = _adapter(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return _completed(argv, "12345;gold-cluster\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    submission = adapter.submit(
        action,
        checkout_root=checkout,
        resources=_resources(),
        placement=ps.SlurmPlacement(
            platform_key="linux-aarch64-sm121", host_class="gb10"
        ),
    )

    assert submission.status == "submitted"
    assert str(submission.job_id) == "12345;gold-cluster"
    assert submission.submission_key is not None
    assert len(calls) == 1
    argv, kwargs = calls[0]
    key = str(action["action_key"])
    assert argv[0] == "/slurm/bin/sbatch"
    assert argv[1:9] == [
        "--parsable",
        "--clusters=gold-cluster",
        "--export=NIL",
        "--no-requeue",
        "--nodes=1",
        "--ntasks=1",
        "--open-mode=append",
        f"--job-name=pqb-{submission.submission_key}",
    ]
    for expected in (
        f"--comment=prismabuild:{submission.submission_key}",
        "--cpus-per-task=8",
        "--mem=65536M",
        "--gpus=1",
        "--constraint=gb10",
        "--partition=gb10",
        "--account=prismaquant",
        "--qos=gold",
        "--time=04:00:00",
        f"--chdir={checkout}",
    ):
        assert expected in argv
    assert argv[argv.index(str(worker)) + 1 :] == [
        "run-local",
        "--require-slurm-initial-start",
        "--action",
        str(tmp_path / "cas" / "requests" / key[:2] / f"{key}.json"),
        "--cas-root",
        str(tmp_path / "cas"),
        "--checkout-root",
        str(checkout),
    ]
    assert kwargs["shell"] is False
    assert kwargs["check"] is False
    assert kwargs["env"] == {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    request = Path(argv[argv.index("--action") + 1])
    assert request.stat().st_mode & 0o222 == 0
    assert json.loads(request.read_text(encoding="utf-8")) == action
    intent_path = (
        tmp_path
        / "cas"
        / "submissions"
        / "v1"
        / key[:2]
        / key
        / "intent.json"
    )
    binding_path = intent_path.with_name("job.json")
    assert intent_path.stat().st_mode & 0o222 == 0
    assert binding_path.stat().st_mode & 0o222 == 0
    assert json.loads(binding_path.read_text(encoding="utf-8"))["job_id"] == (
        "12345;gold-cluster"
    )


def test_cpu_submission_omits_zero_gpu_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(argv, "12345\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter.submit(
        action,
        checkout_root=checkout,
        resources=_resources(gpus=0),
        placement=ps.SlurmPlacement(platform_key=None, host_class=None),
    )
    assert not any(argument.startswith("--gpus=") for argument in calls[0])


def test_submit_refuses_job_id_from_outside_sealed_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: _completed(argv, "12345;other-cluster\n"),
    )
    with pytest.raises(ps.SlurmAdoptionError, match="outside the sealed cluster"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
        )
    _, binding = adapter._submission_paths(str(action["action_key"]))
    assert not binding.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("constraint", "gb10;touch-pwned"),
        ("partition", "--uid=0"),
        ("account", "pq\n--wrap=bad"),
        ("qos", "gold value"),
        ("time_limit", "4 hours"),
    ],
)
def test_resource_fields_refuse_option_or_shell_injection(field: str, value: str):
    with pytest.raises(pb.ActionContractError, match="invalid value"):
        _resources(**{field: value})


def test_submit_refuses_wrong_scope_before_calling_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(
        checkout,
        portability="platform_keyed",
        platform_key="linux-aarch64-sm121",
    )
    adapter, _ = _adapter(tmp_path)
    called = False

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        raise AssertionError("must not call SLURM")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(pb.ActionContractError, match="platform_key"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(
                platform_key="linux-x86-sm89", host_class="gpu"
            ),
        )
    assert not called


def test_host_class_scope_must_be_backed_by_slurm_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(
        checkout, portability="host_class_keyed", host_class="gb10"
    )
    adapter, _ = _adapter(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid placement must not call SLURM")
        ),
    )
    with pytest.raises(pb.ActionContractError, match="partition or constraint"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(constraint="cpu", partition="cpu"),
            placement=ps.SlurmPlacement(
                platform_key="linux-aarch64-sm121", host_class="gb10"
            ),
        )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0",
        "-1",
        "123 extra",
        "123;",
        "123;bad/cluster",
        "1\n2",
        "1;ok;extra",
        "1" * 21,
    ],
)
def test_job_id_parser_refuses_malformed_or_ambiguous_output(raw: str):
    with pytest.raises(ps.SlurmProtocolError, match="malformed"):
        ps.parse_job_id(raw)


def test_published_request_is_first_writer_wins_and_tamper_evident(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    request = ps.publish_action_request(action, cas_root=tmp_path / "cas")
    assert ps.publish_action_request(action, cas_root=tmp_path / "cas") == request
    request.chmod(0o644)
    request.write_text("{}\n", encoding="utf-8")
    with pytest.raises(pb.CASTamperError, match="writable|differs from its action key"):
        ps.publish_action_request(action, cas_root=tmp_path / "cas")


def test_crash_before_sbatch_leaves_ambiguous_intent_and_never_resubmits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    key = str(action["action_key"])

    def crash_before_command(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        intent_path = (
            tmp_path / "cas" / "submissions" / "v1" / key[:2] / key / "intent.json"
        )
        assert intent_path.exists()
        assert not intent_path.with_name("job.json").exists()
        raise KeyboardInterrupt("orchestrator died before sbatch")

    monkeypatch.setattr(subprocess, "run", crash_before_command)
    with pytest.raises(KeyboardInterrupt):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
        )

    calls: list[list[str]] = []

    def accounting_miss(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert argv[0] == "/slurm/bin/sacct"
        assert "--clusters=gold-cluster" in argv
        assert "--allclusters" not in argv
        assert "--duplicates" in argv
        assert "--allocations" in argv
        assert "--starttime=1970-01-01" in argv
        return _completed(argv, "")

    monkeypatch.setattr(subprocess, "run", accounting_miss)
    with pytest.raises(ps.SlurmAdoptionError, match="will not resubmit"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
        )
    assert len(calls) == 1
    assert not any(call[0] == "/slurm/bin/sbatch" for call in calls)


def test_crash_after_sbatch_before_binding_is_discovered_and_adopted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    real_publish = adapter._publish_job_binding
    publish_calls = 0

    def lose_returned_job(
        intent: dict[str, object], job_id: ps.SlurmJobId
    ) -> dict[str, object]:
        nonlocal publish_calls
        publish_calls += 1
        if publish_calls == 1:
            raise RuntimeError("orchestrator died after sbatch accepted the job")
        return real_publish(intent, job_id)

    monkeypatch.setattr(adapter, "_publish_job_binding", lose_returned_job)
    scheduler_calls: list[list[str]] = []

    def first_submit(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        scheduler_calls.append(argv)
        assert argv[0] == "/slurm/bin/sbatch"
        return _completed(argv, "321;gold-cluster\n")

    monkeypatch.setattr(subprocess, "run", first_submit)
    with pytest.raises(RuntimeError, match="after sbatch"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
        )

    intent = adapter._load_submission_intent(str(action["action_key"]))
    assert intent is not None

    def discover(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        scheduler_calls.append(argv)
        assert argv[0] == "/slurm/bin/sacct"
        assert (
            "--format=JobIDRaw%64,Cluster%64,JobName%128,Comment%256,State%64"
            in argv
        )
        return _completed(
            argv,
            f"321|gold-cluster|{intent['job_name']}|{intent['comment']}|RUNNING\n",
        )

    monkeypatch.setattr(subprocess, "run", discover)
    adopted = adapter.submit(
        action,
        checkout_root=checkout,
        resources=_resources(),
        placement=ps.SlurmPlacement(platform_key=None, host_class=None),
    )
    assert adopted.status == "adopted"
    assert str(adopted.job_id) == "321;gold-cluster"
    assert [call[0] for call in scheduler_calls].count("/slurm/bin/sbatch") == 1


def test_replayed_submit_adopts_immutable_binding_without_scheduler_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    calls: list[list[str]] = []

    def submit_once(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(argv, "52\n")

    monkeypatch.setattr(subprocess, "run", submit_once)
    first = adapter.submit(
        action,
        checkout_root=checkout,
        resources=_resources(),
        placement=ps.SlurmPlacement(platform_key=None, host_class=None),
    )
    second = adapter.submit(
        action,
        checkout_root=checkout,
        resources=_resources(),
        placement=ps.SlurmPlacement(platform_key=None, host_class=None),
    )
    assert first.status == "submitted"
    assert second.status == "adopted"
    assert first.job_id == second.job_id == ps.SlurmJobId(52, "gold-cluster")
    assert first.submission_key == second.submission_key
    assert [call[0] for call in calls] == ["/slurm/bin/sbatch"]


def test_existing_intent_refuses_changed_resources_without_second_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_intent(adapter, action, checkout)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("conflicting intent must not query or submit")
        ),
    )
    with pytest.raises(ps.SlurmAdoptionError, match="different sealed submit"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(memory_mib=32768),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
        )


def test_existing_intent_refuses_changed_retry_policy_without_second_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_intent(adapter, action, checkout, max_polls=3)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("conflicting retry policy must not query or submit")
        ),
    )
    with pytest.raises(ps.SlurmAdoptionError, match="different sealed submit"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
            max_polls=4,
            poll_interval_seconds=0.001,
        )


def test_tampered_or_noncanonical_intent_refuses_before_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_intent(adapter, action, checkout)
    intent_path, _ = adapter._submission_paths(str(action["action_key"]))
    altered = dict(intent)
    altered["comment"] = "prismabuild:" + "0" * 64
    body = {key: altered[key] for key in altered if key != "intent_sha256"}
    altered["intent_sha256"] = pb.canonical_sha256(body)
    intent_path.chmod(0o644)
    intent_path.write_text(
        json.dumps(altered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    intent_path.chmod(0o444)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tampered state must fail before scheduler")
        ),
    )
    with pytest.raises(pb.CASTamperError, match="comment is not derived"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
        )


def test_malformed_job_binding_refuses_before_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "81")
    _, binding_path = adapter._submission_paths(str(action["action_key"]))
    binding_path.chmod(0o644)
    binding_path.write_text('{"schema":"not-the-binding-schema"}\n', encoding="utf-8")
    binding_path.chmod(0o444)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("malformed binding must fail before scheduler")
        ),
    )
    with pytest.raises(pb.CASTamperError, match="fields differ"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
        )


def test_ambiguous_duplicate_adoption_rows_fail_without_binding_or_resubmit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_intent(adapter, action, checkout)
    row = f"61||{intent['job_name']}|{intent['comment']}|RUNNING\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: _completed(argv, row + row.replace("61|", "62|", 1)),
    )
    with pytest.raises(ps.SlurmAdoptionError, match="multiple SLURM allocations"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
        )


def test_adoption_refuses_clusterless_accounting_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_intent(adapter, action, checkout)
    row = f"611||{intent['job_name']}|{intent['comment']}|RUNNING\n"
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kwargs: _completed(argv, row)
    )
    with pytest.raises(ps.SlurmAdoptionError, match="outside the sealed.*cluster"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
        )
    _, binding = adapter._submission_paths(str(action["action_key"]))
    assert not binding.exists()
    assert adapter._load_job_binding(intent) is None


def test_terminal_job_is_adopted_and_reported_without_duplicate_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_intent(adapter, action, checkout)
    calls: list[list[str]] = []

    def terminal_rows(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[0] == "/slurm/bin/sacct" and any(
            argument.startswith("--name=") for argument in argv
        ):
            return _completed(
                argv,
                f"71|gold-cluster|{intent['job_name']}|{intent['comment']}|NODE_FAIL\n",
            )
        return _completed(argv, _state_row(intent, 71, "NODE_FAIL"))

    monkeypatch.setattr(subprocess, "run", terminal_rows)
    adopted = adapter.submit(
        action,
        checkout_root=checkout,
        resources=_resources(),
        placement=ps.SlurmPlacement(platform_key=None, host_class=None),
    )
    assert adopted.status == "adopted"
    resolution = adapter.resolve(action, "71")
    assert resolution.status == "failed"
    assert resolution.slurm_state == "NODE_FAIL"
    assert not any(call[0] == "/slurm/bin/sbatch" for call in calls)


def test_wrong_scheduler_identity_blocks_poll_and_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_submission(adapter, action, checkout, "82")
    calls: list[list[str]] = []

    def impersonated_job(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert argv[0] == "/slurm/bin/squeue"
        return _completed(
            argv,
            f"82|RUNNING|{intent['job_name']}|prismabuild:{'f' * 64}\n",
        )

    monkeypatch.setattr(subprocess, "run", impersonated_job)
    with pytest.raises(ps.SlurmAdoptionError, match="durable submission identity"):
        adapter.resolve(action, "82")
    with pytest.raises(ps.SlurmAdoptionError, match="durable submission identity"):
        adapter.cancel(action, "82")
    assert all(call[0] == "/slurm/bin/squeue" for call in calls)


def test_poll_budget_and_pacing_are_durable_across_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(
        adapter,
        action,
        checkout,
        "83",
        max_polls=2,
        poll_interval_seconds=1.0,
    )
    clock = [1_000_000_000]
    sleeps: list[float] = []
    monkeypatch.setattr(ps.time, "time_ns", lambda: clock[0])

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += math.ceil(seconds * 1e9)

    monkeypatch.setattr(ps.time, "sleep", advance)
    assert adapter.claim_poll(action, "83", max_polls=2).polls == 1  # type: ignore[union-attr]
    clock[0] += 250_000_000
    restarted = _restart_adapter(adapter)
    assert restarted.claim_poll(action, "83", max_polls=2).polls == 2  # type: ignore[union-attr]
    assert sleeps == [pytest.approx(0.75)]
    assert restarted.claim_poll(action, "83", max_polls=2) is None


def test_poll_clock_rollback_refuses_without_consuming_a_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(
        adapter,
        action,
        checkout,
        "830",
        max_polls=2,
        poll_interval_seconds=1.0,
    )
    clock = [2_000_000_000]
    monkeypatch.setattr(ps.time, "time_ns", lambda: clock[0])
    adapter.claim_poll(action, "830", max_polls=2)
    clock[0] = 1_999_999_999
    with pytest.raises(ps.SlurmAdoptionError, match="behind the prior"):
        _restart_adapter(adapter).claim_poll(action, "830", max_polls=2)
    assert adapter.retry_progress(action, "830").polls == 1


def test_negative_poll_clock_refuses_before_writing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "8301", max_polls=1)
    monkeypatch.setattr(ps.time, "time_ns", lambda: -1)
    with pytest.raises(ps.SlurmAdoptionError, match="before the Unix epoch"):
        adapter.claim_poll(action, "8301", max_polls=1)
    assert adapter.retry_progress(action, "8301").polls == 0


def test_oversized_poll_clock_refuses_before_writing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "8302", max_polls=1)
    monkeypatch.setattr(ps.time, "time_ns", lambda: ps._MAX_UNIX_NS + 1)
    with pytest.raises(ps.SlurmAdoptionError, match="signed 64-bit"):
        adapter.claim_poll(action, "8302", max_polls=1)
    assert adapter.retry_progress(action, "8302").polls == 0


def test_retry_calls_refuse_policy_drift_and_positive_requeues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "831", max_polls=2)
    monkeypatch.setattr(
        adapter,
        "_run",
        lambda argv: (_ for _ in ()).throw(
            AssertionError("policy refusal must happen before scheduler RPC")
        ),
    )
    with pytest.raises(ps.SlurmAdoptionError, match="sealed submission policy"):
        adapter.claim_poll(action, "831", max_polls=3)
    with pytest.raises(ps.SlurmProtocolError, match="same-job requeue is disabled"):
        adapter.requeue(action, "831", max_requeues=1)


def test_submit_rejects_positive_requeue_and_recompute_before_scheduler_or_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported policy must not reach Slurm")
        ),
    )
    kwargs = {
        "checkout_root": checkout,
        "resources": _resources(),
        "placement": ps.SlurmPlacement(platform_key=None, host_class=None),
    }
    with pytest.raises(pb.ActionContractError, match="max_requeues must be zero"):
        adapter.submit(action, max_requeues=1, **kwargs)
    with pytest.raises(pb.ActionContractError, match="recompute is disabled"):
        adapter.submit(action, recompute=True, **kwargs)
    with pytest.raises(pb.ActionContractError, match="must not exceed 86400"):
        adapter.submit(action, poll_interval_seconds=86_401, **kwargs)
    assert not adapter._submission_directory(str(action["action_key"])).exists()


def test_durable_poll_count_beyond_sealed_maximum_is_tamper(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_submission(adapter, action, checkout, "832", max_polls=1)
    adapter.claim_poll(action, "832", max_polls=1)
    binding = adapter._load_job_binding(intent)
    assert binding is not None
    extra = adapter._transition_body(
        intent=intent,
        binding=binding,
        kind="poll",
        attempt=0,
        ordinal=2,
        claimed_unix_ns=time.time_ns(),
    )
    adapter._publish_state_file(
        adapter._transition_directory(str(action["action_key"]), "polls")
        / "00000000-00000002.json",
        extra,
    )
    with pytest.raises(pb.CASTamperError, match="exceeds the sealed maximum"):
        _restart_adapter(adapter).retry_progress(action, "832")


def test_durable_poll_timestamp_beyond_signed_range_is_tamper(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_submission(adapter, action, checkout, "8321", max_polls=1)
    binding = adapter._load_job_binding(intent)
    assert binding is not None
    transition = adapter._transition_body(
        intent=intent,
        binding=binding,
        kind="poll",
        attempt=0,
        ordinal=1,
        claimed_unix_ns=ps._MAX_UNIX_NS + 1,
    )
    adapter._publish_state_file(
        adapter._transition_directory(str(action["action_key"]), "polls")
        / "00000000-00000001.json",
        transition,
    )
    with pytest.raises(pb.CASTamperError, match="signed 64-bit"):
        _restart_adapter(adapter).retry_progress(action, "8321")


def test_durable_state_huge_json_integer_fails_in_tamper_vocabulary(tmp_path: Path):
    state = tmp_path / "state.json"
    state.write_bytes(b'{"oversized":' + b"9" * 10_000 + b"}\n")
    state.chmod(0o444)
    with pytest.raises(pb.CASTamperError, match="not strict UTF-8 JSON"):
        ps.SlurmAdapter._read_state_file(state, where="hostile durable state")


def test_durable_state_file_size_is_bounded_before_reading(tmp_path: Path):
    state = tmp_path / "state.json"
    state.touch()
    with state.open("r+b") as handle:
        handle.truncate(ps._MAX_STATE_BYTES + 1)
    state.chmod(0o444)
    with pytest.raises(pb.CASTamperError, match="state byte bound"):
        ps.SlurmAdapter._read_state_file(state, where="oversized durable state")


def test_durable_state_chmod_during_read_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state = tmp_path / "state.json"
    state.write_bytes(b"{}\n")
    state.chmod(0o444)
    real_read = ps.os.read
    changed = False

    def read_then_chmod(descriptor: int, count: int) -> bytes:
        nonlocal changed
        chunk = real_read(descriptor, count)
        if chunk and not changed:
            changed = True
            state.chmod(0o644)
        return chunk

    monkeypatch.setattr(ps.os, "read", read_then_chmod)
    with pytest.raises(pb.CASTamperError, match="changed while read"):
        ps.SlurmAdapter._read_state_file(state, where="racing durable state")


def test_durable_requeue_count_beyond_sealed_maximum_is_tamper(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_submission(adapter, action, checkout, "833")
    binding = adapter._load_job_binding(intent)
    assert binding is not None
    transition = adapter._transition_body(
        intent=intent,
        binding=binding,
        kind="requeue",
        attempt=0,
        ordinal=1,
        claimed_unix_ns=time.time_ns(),
    )
    adapter._publish_state_file(
        adapter._transition_directory(str(action["action_key"]), "requeues")
        / "00000001.json",
        transition,
    )
    with pytest.raises(pb.CASTamperError, match="requeue.*count exceeds"):
        _restart_adapter(adapter).retry_progress(action, "833")


def test_retry_history_rejects_unbounded_stale_temp_flood(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "8331")
    directory = adapter._transition_directory(str(action["action_key"]), "requeues")
    directory.mkdir(parents=True)
    for ordinal in range(65):
        (directory / f".{ordinal}.tmp").write_bytes(b"")
    with pytest.raises(pb.CASTamperError, match="too many stale.*temp"):
        adapter.retry_progress(action, "8331")


def test_slurm_state_rejects_symlinked_ancestor_component(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "834")
    submission = adapter._submission_directory(str(action["action_key"]))
    outside = tmp_path / "outside-transitions"
    outside.mkdir()
    (submission / "transitions").symlink_to(outside, target_is_directory=True)
    with pytest.raises(pb.CASTamperError, match="ancestor is not a real directory"):
        adapter.retry_progress(action, "834")


def test_action_request_publish_rejects_symlinked_cas_ancestor(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    cas_root = tmp_path / "cas"
    cas_root.mkdir()
    outside = tmp_path / "outside-requests"
    outside.mkdir()
    (cas_root / "requests").symlink_to(outside, target_is_directory=True)
    with pytest.raises(pb.CASTamperError, match="without following links"):
        ps.publish_action_request(action, cas_root=cas_root)


def test_directory_creation_uses_dirfd_when_parent_path_is_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    root = anchor / "cas"
    target = root / "requests" / "aa"
    parked = tmp_path / "parked-anchor"
    outside = tmp_path / "outside-create-race"
    outside.mkdir()
    real_mkdir = os.mkdir
    calls: list[tuple[object, int | None]] = []
    swapped = False

    def mkdir_and_swap(
        path: object, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> None:
        nonlocal swapped
        calls.append((path, dir_fd))
        assert dir_fd is not None, "descendant mkdir must be dirfd-anchored"
        real_mkdir(path, mode, dir_fd=dir_fd)  # type: ignore[arg-type]
        if path == "cas" and not swapped:
            swapped = True
            anchor.rename(parked)
            real_mkdir(anchor, 0o755)
            (anchor / "cas").symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(ps.os, "mkdir", mkdir_and_swap)
    with pytest.raises(pb.CASTamperError, match="without following links"):
        ps._ensure_real_directory(target, root=root, where="hostile create race")

    assert [path for path, _ in calls] == ["cas", "requests", "aa"]
    assert all(directory_fd is not None for _, directory_fd in calls)
    assert (parked / "cas" / "requests" / "aa").is_dir()
    assert not (outside / "requests").exists()


def test_atomic_publish_syncs_readonly_mode_after_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    directory = tmp_path / "publish"
    directory.mkdir()
    target = directory / "claim.json"
    real_fchmod = os.fchmod
    real_fsync = os.fsync
    events: list[tuple[str, int]] = []

    def recording_fchmod(descriptor: int, mode: int) -> None:
        events.append(("fchmod", descriptor))
        real_fchmod(descriptor, mode)

    def recording_fsync(descriptor: int) -> None:
        events.append(("fsync", descriptor))
        real_fsync(descriptor)

    monkeypatch.setattr(ps.os, "fchmod", recording_fchmod)
    monkeypatch.setattr(ps.os, "fsync", recording_fsync)
    assert ps._atomic_publish_nofollow(target, b"{}\n", where="test claim")

    assert events[0][0] == "fchmod"
    assert events[1] == ("fsync", events[0][1])
    assert stat.S_IMODE(target.stat().st_mode) == 0o444


def test_state_publish_rejects_parent_swapped_to_symlink_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "8341")
    parent = adapter._mutation_directory(str(action["action_key"]))
    target = parent / "00000001.json"
    outside = tmp_path / "outside-swap"
    outside.mkdir()
    parked = tmp_path / "parked-mutations"
    real_ensure = ps._ensure_real_directory

    def ensure_then_swap(path: Path, *, root: Path, where: str) -> None:
        real_ensure(path, root=root, where=where)
        if path == parent:
            path.rename(parked)
            path.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(ps, "_ensure_real_directory", ensure_then_swap)
    with pytest.raises(pb.CASTamperError, match="without following links"):
        adapter._publish_state_file(target, {"hostile": True})
    assert not (outside / target.name).exists()


def test_state_read_rejects_parent_swapped_to_symlink_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "8342")
    intent_path, _ = adapter._submission_paths(str(action["action_key"]))
    parent = intent_path.parent
    parked = tmp_path / "parked-submission"
    outside = tmp_path / "outside-read"
    outside.mkdir()
    hostile = outside / intent_path.name
    hostile.write_text("{}\n", encoding="utf-8")
    hostile.chmod(0o444)
    real_chain = ps._real_directory_chain_exists
    swapped = [False]

    def validate_then_swap(path: Path, *, where: str) -> bool:
        result = real_chain(path, where=where)
        if path == parent and result and not swapped[0]:
            swapped[0] = True
            path.rename(parked)
            path.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(ps, "_real_directory_chain_exists", validate_then_swap)
    with pytest.raises(pb.CASTamperError, match="without following links"):
        adapter._read_state_file(intent_path, where="hostile read race")


def test_clusterless_scheduler_arguments_are_forced_local():
    assert ps.SlurmAdapter._cluster_args(ps.SlurmJobId(7)) == ["--local"]


def test_slurm_worker_restart_guard_refuses_before_task_execution():
    pb._require_slurm_initial_start(
        {"SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "0"}
    )
    with pytest.raises(pb.ActionContractError, match="not authorized"):
        pb._require_slurm_initial_start(
            {"SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "1"}
        )
    with pytest.raises(pb.ActionContractError, match="malformed"):
        pb._require_slurm_initial_start(
            {"SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "admin"}
        )
    for malformed in ("00", "01", "1" * 10_000, 1):
        with pytest.raises(pb.ActionContractError, match="malformed"):
            pb._require_slurm_initial_start(  # type: ignore[arg-type]
                {"SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": malformed}
            )
    with pytest.raises(pb.ActionContractError, match="positive numeric"):
        pb._require_slurm_initial_start(  # type: ignore[arg-type]
            {"SLURM_JOB_ID": 123, "SLURM_RESTART_COUNT": "0"}
        )


def test_slurm_worker_cli_blocks_restarted_allocation_before_run_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    request = tmp_path / "action.json"
    request.write_text(json.dumps(action), encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    monkeypatch.setattr(
        pb,
        "run_local_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("restarted worker must not reach run_local_action")
        ),
    )
    with pytest.raises(pb.ActionContractError, match="not authorized"):
        pb.main(
            [
                "run-local",
                "--require-slurm-initial-start",
                "--action",
                str(request),
                "--cas-root",
                str(tmp_path / "cas"),
                "--checkout-root",
                str(checkout),
            ]
        )


def test_slurm_worker_cli_rejects_recompute_even_on_initial_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    request = tmp_path / "action.json"
    request.write_text(json.dumps(action), encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    monkeypatch.setattr(
        pb,
        "run_local_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Slurm recompute must not reach run_local_action")
        ),
    )
    with pytest.raises(pb.ActionContractError, match="does not permit recompute"):
        pb.main(
            [
                "run-local",
                "--require-slurm-initial-start",
                "--recompute",
                "--action",
                str(request),
                "--cas-root",
                str(tmp_path / "cas"),
                "--checkout-root",
                str(checkout),
            ]
        )


def test_slurm_worker_cli_requires_canonical_anchored_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    cas_root = tmp_path / "cas"
    canonical = ps.publish_action_request(action, cas_root=cas_root)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    calls: list[dict[str, object]] = []

    def run_local(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"status": "test-only"}

    monkeypatch.setattr(pb, "run_local_action", run_local)
    assert pb.main(
        [
            "run-local",
            "--require-slurm-initial-start",
            "--action",
            str(canonical),
            "--cas-root",
            str(cas_root),
            "--checkout-root",
            str(checkout),
        ]
    ) == 0
    assert len(calls) == 1
    assert calls[0]["recompute"] is False

    alias = tmp_path / "copied-action.json"
    alias.write_bytes(canonical.read_bytes())
    alias.chmod(0o444)
    with pytest.raises(pb.ActionContractError, match="canonical CAS request"):
        pb.main(
            [
                "run-local",
                "--require-slurm-initial-start",
                "--action",
                str(alias),
                "--cas-root",
                str(cas_root),
                "--checkout-root",
                str(checkout),
            ]
        )
    assert len(calls) == 1


def test_corrupt_retry_counter_refuses_progress_replay(
    tmp_path: Path
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "86", max_polls=3)
    adapter.claim_poll(action, "86", max_polls=3)
    counter = (
        adapter._transition_directory(str(action["action_key"]), "polls")
        / "00000000-00000001.json"
    )
    value = json.loads(counter.read_text(encoding="utf-8"))
    value["ordinal"] = 2
    body = {key: value[key] for key in value if key != "transition_sha256"}
    value["transition_sha256"] = pb.canonical_sha256(body)
    counter.chmod(0o644)
    counter.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    counter.chmod(0o444)
    with pytest.raises(pb.CASTamperError, match="ordinal differs"):
        _restart_adapter(adapter).retry_progress(action, "86")


def test_completed_without_receipt_is_failure_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_submission(adapter, action, checkout, "91")
    outputs = iter(["", _state_row(intent, 91, "COMPLETED")])

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(argv, next(outputs))

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = adapter.resolve(action, "91")
    assert result.status == "failed"
    assert result.slurm_state == "COMPLETED"
    assert "no valid CAS receipt" in result.reason


def test_sacct_state_query_requests_untruncated_identity_fields_and_strips_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_submission(adapter, action, checkout, "92")
    calls: list[list[str]] = []

    def padded_accounting(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[0] == "/slurm/bin/squeue":
            return _completed(argv, "")
        return _completed(
            argv,
            " 92 | COMPLETED | "
            f"{intent['job_name']} | {intent['comment']} \n",
        )

    monkeypatch.setattr(subprocess, "run", padded_accounting)
    assert adapter.resolve(action, "92").status == "failed"
    assert "--format=JobIDRaw%64,State%64,JobName%128,Comment%256" in calls[1]
    assert "--duplicates" in calls[1]


def test_new_job_absent_from_both_controllers_is_bounded_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "91")

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(argv, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = adapter.resolve(action, "91")
    assert result.status == "pending"
    assert result.slurm_state == "NOT_VISIBLE"
    assert "not yet visible" in result.reason


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("PENDING", "pending"),
        ("EXPEDITING", "pending"),
        ("POWER_UP_NODE", "pending"),
        ("REQUEUED", "pending"),
        ("SPECIAL_EXIT", "pending"),
        ("UPDATE_DB", "pending"),
        ("RUNNING", "running"),
        ("COMPLETING", "running"),
        ("STOPPED", "running"),
        ("CANCELLED by 1000", "cancelled"),
        ("LAUNCH_FAILED", "failed"),
        ("RECONFIG_FAIL", "failed"),
        ("OUT_OF_MEMORY", "failed"),
    ],
)
def test_state_normalization_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    expected: str,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_submission(adapter, action, checkout, "77")

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(argv, _state_row(intent, 77, state))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert adapter.resolve(action, "77").status == expected


@pytest.mark.parametrize("case", ["unknown", "ambiguous", "wrong_job"])
def test_unknown_ambiguous_or_wrong_job_state_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_submission(adapter, action, checkout, "77")
    if case == "unknown":
        output = _state_row(intent, 77, "MYSTERY")
    elif case == "ambiguous":
        output = _state_row(intent, 77, "RUNNING") + _state_row(
            intent, 77, "PENDING"
        )
    else:
        output = _state_row(intent, 88, "RUNNING")

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(argv, output)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ps.SlurmProtocolError):
        adapter.resolve(action, "77")


def test_verified_receipt_is_success_even_before_slurm_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(
        checkout, portability="host_class_keyed", host_class="gb10"
    )
    output = tmp_path / "result"
    output.write_bytes(b"canonical")
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "sparky")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gb10")
    monkeypatch.setattr(
        pb,
        "_verify_slurm_process_membership",
        lambda job_id: f"/slurm/job_{job_id}/step_batch",
    )
    attestation = _attestation(checkout, action, tmp_path / "cas")
    receipt, _ = cas.publish_result(
        action,
        output,
        attestation=attestation,
    )
    adapter, _ = _adapter(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("valid receipt must avoid scheduler query")

    monkeypatch.setattr(subprocess, "run", forbidden)
    result = adapter.resolve(action, "123")
    assert result.status == "succeeded"
    assert result.receipt == receipt
    assert result.payload_path is not None
    assert result.payload_path.read_bytes() == b"canonical"


def test_submit_is_cache_hit_noop_after_verified_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    output = tmp_path / "result"
    output.write_bytes(b"canonical")
    receipt, _ = pb.PrismaBuildCAS(tmp_path / "cas").publish_result(
        action,
        output,
        attestation=_attestation(checkout, action, tmp_path / "cas"),
    )
    adapter, _ = _adapter(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache hit must not submit")
        ),
    )
    submission = adapter.submit(
        action,
        checkout_root=checkout,
        resources=_resources(),
        placement=ps.SlurmPlacement(
            platform_key=None, host_class=None
        ),
    )
    assert submission.status == "cache_hit"
    assert submission.job_id is None
    assert submission.receipt == receipt


def test_wrong_scope_self_consistent_receipt_is_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(
        checkout, portability="host_class_keyed", host_class="gb10"
    )
    output = tmp_path / "result"
    output.write_bytes(b"canonical")
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "sparky")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gb10")
    monkeypatch.setattr(
        pb,
        "_verify_slurm_process_membership",
        lambda job_id: f"/slurm/job_{job_id}/step_batch",
    )
    attestation = _attestation(checkout, action, tmp_path / "cas")
    receipt, _ = cas.publish_result(
        action,
        output,
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
    receipt_path = cas._receipt_path(str(action["action_key"]))
    receipt_path.chmod(0o644)
    receipt_path.write_text(
        json.dumps(altered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)
    adapter, _ = _adapter(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must fail before scheduler query")
        ),
    )
    with pytest.raises(pb.CASTamperError, match="host_class"):
        adapter.resolve(action, "123")


def test_cancel_uses_exact_cluster_scoped_argv_and_requeue_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "991;gold-cluster")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(argv)

    monkeypatch.setattr(subprocess, "run", fake_run)
    states = iter([("RUNNING", "running"), ("RUNNING", "running")])
    monkeypatch.setattr(adapter, "_query_state", lambda *args, **kwargs: next(states))
    assert adapter.cancel(action, "991;gold-cluster")
    with pytest.raises(ps.SlurmProtocolError, match="same-job requeue is disabled"):
        adapter.requeue(action, "991;gold-cluster", max_requeues=1)
    assert calls == [["/slurm/bin/scancel", "--clusters=gold-cluster", "991"]]


def test_cancel_rpc_crash_leaves_final_claim_and_is_never_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "992")
    monkeypatch.setattr(
        adapter, "_query_state", lambda *args, **kwargs: ("RUNNING", "running")
    )
    calls: list[list[str]] = []

    def crash(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        raise KeyboardInterrupt("scancel acceptance is unknown")

    monkeypatch.setattr(adapter, "_run", crash)
    with pytest.raises(KeyboardInterrupt):
        adapter.cancel(action, "992")
    restarted = _restart_adapter(adapter)
    monkeypatch.setattr(
        restarted,
        "_run",
        lambda argv: (_ for _ in ()).throw(
            AssertionError("an ambiguous scancel must never be replayed")
        ),
    )
    with pytest.raises(ps.SlurmAdoptionError, match="cancel claim already exists"):
        restarted.cancel(action, "992")
    assert calls == [["/slurm/bin/scancel", "--clusters=gold-cluster", "992"]]


def test_crash_immediately_after_cancel_append_never_reaches_scancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "9921")
    monkeypatch.setattr(
        adapter, "_query_state", lambda *args, **kwargs: ("RUNNING", "running")
    )
    real_claim = adapter._claim_mutation

    def append_then_die(**kwargs: object) -> dict[str, object]:
        real_claim(**kwargs)  # type: ignore[arg-type]
        raise KeyboardInterrupt("died after durable cancel claim")

    monkeypatch.setattr(adapter, "_claim_mutation", append_then_die)
    monkeypatch.setattr(
        adapter,
        "_run",
        lambda argv: (_ for _ in ()).throw(
            AssertionError("crash after append must not reach scancel")
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        adapter.cancel(action, "9921")
    restarted = _restart_adapter(adapter)
    with pytest.raises(ps.SlurmAdoptionError, match="cancel claim already exists"):
        restarted.cancel(action, "9921")


def test_crash_before_cancel_append_allows_one_safe_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "9922")
    monkeypatch.setattr(
        adapter, "_query_state", lambda *args, **kwargs: ("RUNNING", "running")
    )
    monkeypatch.setattr(
        adapter,
        "_claim_mutation",
        lambda **kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("died before durable cancel claim")
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        adapter.cancel(action, "9922")
    restarted = _restart_adapter(adapter)
    monkeypatch.setattr(
        restarted, "_query_state", lambda *args, **kwargs: ("RUNNING", "running")
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        restarted,
        "_run",
        lambda argv: (calls.append(list(argv)) or _completed(list(argv))),
    )
    assert restarted.cancel(action, "9922")
    assert calls == [["/slurm/bin/scancel", "--clusters=gold-cluster", "9922"]]


def test_concurrent_cancel_claim_has_one_winner_and_one_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    first, _ = _adapter(tmp_path)
    _record_submission(first, action, checkout, "993")
    second = _restart_adapter(first)
    barrier = threading.Barrier(2)
    calls: list[list[str]] = []
    calls_lock = threading.Lock()

    for adapter in (first, second):
        query_count = [0]

        def query(*args: object, _count=query_count, **kwargs: object):
            _count[0] += 1
            if _count[0] == 1:
                barrier.wait(timeout=5)
            return "RUNNING", "running"

        monkeypatch.setattr(adapter, "_query_state", query)

        def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
            with calls_lock:
                calls.append(list(argv))
            return _completed(list(argv))

        monkeypatch.setattr(adapter, "_run", run)

    def cancel(adapter: ps.SlurmAdapter) -> bool | Exception:
        try:
            return adapter.cancel(action, "993")
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(cancel, (first, second)))
    assert sum(outcome is True for outcome in outcomes) == 1
    assert sum(isinstance(outcome, ps.SlurmAdoptionError) for outcome in outcomes) == 1
    assert calls == [["/slurm/bin/scancel", "--clusters=gold-cluster", "993"]]


def test_cancel_is_final_in_unified_mutation_journal(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_submission(adapter, action, checkout, "994")
    binding = adapter._load_job_binding(intent)
    assert binding is not None
    adapter._claim_mutation(intent=intent, binding=binding, kind="cancel")
    body = {
        "schema": ps.SLURM_MUTATION_SCHEMA_V1,
        "action_key": intent["action_key"],
        "submission_key": intent["submission_key"],
        "job_id": binding["job_id"],
        "kind": "requeue",
        "ordinal": 2,
    }
    mutation = {**body, "mutation_sha256": pb.canonical_sha256(body)}
    adapter._publish_state_file(
        adapter._mutation_directory(str(action["action_key"])) / "00000002.json",
        mutation,
    )
    with pytest.raises(pb.CASTamperError, match="mutation count exceeds"):
        adapter._load_mutations(intent=intent, binding=binding)


def test_internal_requeue_mutation_cannot_poison_zero_budget_lineage(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    intent = _record_submission(adapter, action, checkout, "995")
    binding = adapter._load_job_binding(intent)
    assert binding is not None
    with pytest.raises(ps.SlurmProtocolError, match="exceed the sealed maximum"):
        adapter._claim_mutation(intent=intent, binding=binding, kind="requeue")
    assert not adapter._mutation_directory(str(action["action_key"])).exists()


def test_requeue_is_noop_after_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    output = tmp_path / "result"
    output.write_bytes(b"done")
    pb.PrismaBuildCAS(tmp_path / "cas").publish_result(
        action,
        output,
        attestation=_attestation(checkout, action, tmp_path / "cas"),
    )
    adapter, _ = _adapter(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no retry")),
    )
    assert adapter.requeue(action, "991", max_requeues=1) is False
    with pytest.raises(pb.ActionContractError, match="recompute is disabled"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(platform_key=None, host_class=None),
            recompute=True,
        )


def test_absent_slurm_binary_and_nonzero_command_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    _record_submission(adapter, action, checkout, "1")
    monkeypatch.setattr(
        adapter, "_query_state", lambda *args, **kwargs: ("RUNNING", "running")
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("absent")),
    )
    with pytest.raises(ps.SlurmUnavailableError, match="cannot execute"):
        adapter.cancel(action, "1")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 2, stdout="", stderr="permission denied"
        ),
    )
    with pytest.raises(ps.SlurmAdoptionError, match="cancel claim already exists"):
        adapter.cancel(action, "1")
    with pytest.raises(ps.SlurmCommandError, match="permission denied"):
        adapter._run(["/slurm/bin/scancel", "--clusters=gold-cluster", "1"])


def test_worker_script_must_be_real_and_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, worker = _adapter(tmp_path)
    worker.chmod(0o644)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no submit")),
    )
    with pytest.raises(pb.ActionContractError, match="must be executable"):
        adapter.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(
                platform_key=None, host_class=None
            ),
        )

    worker.chmod(0o555)
    link = tmp_path / "worker-link"
    link.symlink_to(worker)
    linked = ps.SlurmAdapter(
        cas_root=tmp_path / "cas2",
        log_root=tmp_path / "logs2",
        worker_script=link,
        cluster="gold-cluster",
        sbatch="/slurm/bin/sbatch",
    )
    with pytest.raises(pb.ActionContractError, match="non-symlink"):
        linked.submit(
            action,
            checkout_root=checkout,
            resources=_resources(),
            placement=ps.SlurmPlacement(
                platform_key=None, host_class=None
            ),
        )
