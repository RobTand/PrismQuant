from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
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
    toolchain = {"python": "3.12"}
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
            sbatch="/slurm/bin/sbatch",
            squeue="/slurm/bin/squeue",
            sacct="/slurm/bin/sacct",
            scancel="/slurm/bin/scancel",
            scontrol="/slurm/bin/scontrol",
        ),
        worker,
    )


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
    assert len(calls) == 1
    argv, kwargs = calls[0]
    key = str(action["action_key"])
    assert argv[0] == "/slurm/bin/sbatch"
    assert argv[1:8] == [
        "--parsable",
        "--export=NONE",
        "--requeue",
        "--nodes=1",
        "--ntasks=1",
        "--open-mode=append",
        f"--job-name=pq-{key[:24]}",
    ]
    for expected in (
        f"--comment=prismabuild:{key}",
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
    ["", "0", "-1", "123 extra", "123;", "123;bad/cluster", "1\n2", "1;ok;extra"],
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
    with pytest.raises(pb.CASTamperError, match="differs from its action key"):
        ps.publish_action_request(action, cas_root=tmp_path / "cas")


def test_completed_without_receipt_is_failure_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    outputs = iter(["", "91|COMPLETED\n"])

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(argv, next(outputs))

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = adapter.resolve(action, "91")
    assert result.status == "failed"
    assert result.slurm_state == "COMPLETED"
    assert "no valid CAS receipt" in result.reason


def test_new_job_absent_from_both_controllers_is_bounded_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)

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
        ("REQUEUED", "pending"),
        ("RUNNING", "running"),
        ("COMPLETING", "running"),
        ("CANCELLED by 1000", "cancelled"),
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

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(argv, f"77|{state}\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert adapter.resolve(action, "77").status == expected


@pytest.mark.parametrize("output", ["77|MYSTERY\n", "77|RUNNING\n77|PENDING\n", "88|RUNNING\n"])
def test_unknown_ambiguous_or_wrong_job_state_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: str,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)

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
    receipt_path = (
        tmp_path
        / "cas"
        / "actions"
        / str(action["action_key"])[:2]
        / f"{action['action_key']}.json"
    )
    receipt_path.chmod(0o644)
    receipt_path.write_text(
        json.dumps(altered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
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


def test_cancel_and_requeue_use_exact_cluster_scoped_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    adapter, _ = _adapter(tmp_path)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(argv)

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter.cancel("991;gold-cluster")
    assert adapter.requeue(action, "991;gold-cluster")
    assert calls == [
        ["/slurm/bin/scancel", "--clusters=gold-cluster", "991"],
        ["/slurm/bin/scontrol", "--clusters=gold-cluster", "requeue", "991"],
    ]


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
    assert adapter.requeue(action, "991") is False


def test_absent_slurm_binary_and_nonzero_command_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter, _ = _adapter(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("absent")),
    )
    with pytest.raises(ps.SlurmUnavailableError, match="cannot execute"):
        adapter.cancel("1")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 2, stdout="", stderr="permission denied"
        ),
    )
    with pytest.raises(ps.SlurmCommandError, match="permission denied"):
        adapter.cancel("1")


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
