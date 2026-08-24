"""Deterministic two-host coordinator for the RTX4090 FP8-CB campaign.

The numerical work remains owned by :mod:`sample_parallel_probe`,
:mod:`incremental_probe`, and :mod:`rtx4090_fp8_burn`.  This module owns only
their closed two-host schedule: fixed arguments, immutable Docker mounts,
barriers, durable coordinator state, and transport receipts.  A manifest can
select hosts and absolute roots; it cannot supply executable commands.

Live transport wiring is deliberately fail-closed until the transport can
return one durable job receipt *and* GPU-utilization samples for every
GPU-bearing assignment.  Tests inject that interface to exercise the complete
state machine without launching a process.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

from prismaquant.cluster_campaign_contract import (
    CANONICAL_CONTAINER_PATHS,
    STAGE_DAG,
    StageAssignment,
    canonical_sha256,
    complete_assignment,
    initial_campaign_state,
    next_ready_assignments,
    parse_campaign_manifest,
    validate_campaign_manifest,
    validate_campaign_state,
)
from prismaquant.cluster_transport import (
    JobReceipt,
    RunRequest,
    TelemetrySnapshot,
    summarize_utilization,
)


COMMAND_PLAN_SCHEMA = "prismaquant.rtx4090_two_host_campaign.command_plan.v1"
COMMAND_SCHEMA = "prismaquant.rtx4090_two_host_campaign.command.v1"
EXECUTION_RECEIPT_SCHEMA = (
    "prismaquant.rtx4090_two_host_campaign.execution_receipt.v1"
)
STATUS_SCHEMA = "prismaquant.rtx4090_two_host_campaign.status.v1"
IMAGE_REPOSITORY = "eugr/spark-vllm"
SOURCE_BOOTSTRAP = "/pq/tools/prismaquant_source_bootstrap.py"
SNAPSHOT_MANIFEST = "/pq/.prismaquant-runtime-snapshot.json"
STATE_FILE = "campaign-state.json"
PLAN_FILE = "command-plan.json"
MANIFEST_FILE = "campaign-manifest.json"
RECEIPT_DIR = "receipts"
LOCK_FILE = "campaign.lock"

_HOST_ENV = (
    ("LANG", "C.UTF-8"),
    ("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
    ("PYTHONNOUSERSITE", "1"),
    ("PYTHONSAFEPATH", "1"),
)

_HEX64 = re.compile(r"[0-9a-f]{64}")
_GPU_STAGES = frozenset({
    "coordinator_source_identity",
    "worker_source_identity",
    "sample_ce",
    "sample_fisher",
    "measure_burn",
})
_RESUMABLE_FLAG_STAGES = frozenset({"measure_burn", "allocate"})
_STAGE_DEPENDENCIES = {
    spec.stage: frozenset(spec.dependencies) for spec in STAGE_DAG
}
_EXECUTION_RECEIPT_KEYS = frozenset({
    "schema",
    "campaign_identity_sha256",
    "host_identity_sha256",
    "producer_identity_sha256",
    "work_id",
    "stage",
    "host_id",
    "assignment_index",
    "command_identity_sha256",
    "request_sha256",
    "dependency_receipt_sha256s",
    "job",
    "gpu_bearing",
    "telemetry",
    "telemetry_sha256",
    "telemetry_summary",
    "identity_sha256",
})
_BOOTSTRAP_SCRIPT = r"""
set -euo pipefail
module="$1"
shift
python3 -P -B -s /pq/tools/prismaquant_runtime_snapshot.py verify \
  --snapshot /pq \
  --expected-commit "$PQ_EXPECTED_COMMIT" \
  --expected-tree "$PQ_EXPECTED_TREE" \
  --expected-closure-sha256 "$PQ_EXPECTED_CLOSURE" >/dev/null
exec env -u PYTHONPATH \
  PQ_RUNTIME_PRISMAQUANT_ROOT=/pq \
  python3 -P -B -s /pq/tools/prismaquant_source_bootstrap.py \
    run-module --source-root /pq "$module" "$@"
""".strip()


class RTX4090TwoHostCampaignError(RuntimeError):
    """The fixed campaign plan, state, or transport receipt is invalid."""


class CampaignTransport(Protocol):
    """Execution seam required by :class:`CampaignCoordinator`.

    The shared transports intentionally expose lower-level ``start/status``
    operations.  A live adapter must combine those with periodic telemetry and
    return ``(JobReceipt, samples)`` here.  Merely offering blocking ``run`` is
    insufficient because it cannot prove utilization over the job interval.
    """

    def run_with_telemetry(
        self, request: RunRequest,
    ) -> tuple[object, Sequence[object]]:
        ...


@dataclass(frozen=True)
class CampaignCommand:
    work_id: str
    stage: str
    host_id: str
    assignment_index: int
    worker_index: int
    module: str | None
    module_argv: tuple[str, ...]
    argv: tuple[str, ...]
    resume_argv: tuple[str, ...] | None
    env: tuple[tuple[str, str], ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    gpu_bearing: bool

    def body(self) -> dict[str, object]:
        return {
            "schema": COMMAND_SCHEMA,
            "work_id": self.work_id,
            "stage": self.stage,
            "host_id": self.host_id,
            "assignment_index": self.assignment_index,
            "worker_index": self.worker_index,
            "module": self.module,
            "module_argv": list(self.module_argv),
            "argv": list(self.argv),
            "resume_argv": (
                list(self.resume_argv) if self.resume_argv is not None else None
            ),
            "env": [[name, value] for name, value in self.env],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "gpu_bearing": self.gpu_bearing,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.body())

    def to_dict(self) -> dict[str, object]:
        body = self.body()
        return {**body, "identity_sha256": self.identity_sha256}


def _host_by_id(
    manifest: Mapping[str, object], host_id: str,
) -> Mapping[str, object]:
    for raw in manifest["hosts"]:  # type: ignore[index]
        if isinstance(raw, Mapping) and raw.get("id") == host_id:
            return raw
    raise RTX4090TwoHostCampaignError(f"manifest has no host {host_id!r}")


def _worker_index(
    manifest: Mapping[str, object], host_id: str,
) -> int:
    """Return the stable partition/stripe index, never the DAG index."""

    host_ids = [
        str(host["id"])
        for host in manifest["hosts"]  # type: ignore[index]
    ]
    try:
        index = host_ids.index(host_id)
    except ValueError as exc:  # validated manifests make this unreachable
        raise RTX4090TwoHostCampaignError(
            f"manifest has no worker index for host {host_id!r}"
        ) from exc
    if host_ids != sorted(host_ids) or index not in (0, 1):
        raise RTX4090TwoHostCampaignError(
            "campaign host order cannot define exact partitions 0 and 1"
        )
    return index


def _safe_mount_source(value: object, *, field: str) -> str:
    text = str(value)
    if not text.startswith("/") or any(char in text for char in (",", "\n", "\0")):
        raise RTX4090TwoHostCampaignError(
            f"host root {field} cannot be represented by Docker --mount"
        )
    return text


def _container_environment(
    manifest: Mapping[str, object], *, source_cache: bool = True,
) -> tuple[tuple[str, str], ...]:
    producer = manifest["producer"]  # type: ignore[index]
    assert isinstance(producer, Mapping)
    values = {
        "HOME": "/worker-state/home",
        "XDG_CACHE_HOME": "/worker-state/cache",
        "TMPDIR": "/worker-state/tmp",
        "TORCHINDUCTOR_CACHE_DIR": "/worker-state/torchinductor",
        "TRITON_CACHE_DIR": "/worker-state/triton",
        "PRISMAQUANT_PRODUCER_SNAPSHOT_ROOT": "/pq",
        "PRISMAQUANT_PRODUCER_IMAGE_DIGEST": str(producer["image_digest"]),
        "PRISMAQUANT_CB_ENCODE_COMPILE": "1",
        "PRISMAQUANT_CB_ATOM_COMPILE": "1",
        "PRISMAQUANT_CB_COMPILE_FAIL_CLOSED": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PQ_EXPECTED_COMMIT": str(producer["commit"]),
        "PQ_EXPECTED_TREE": str(producer["tree"]),
        "PQ_EXPECTED_CLOSURE": str(producer["snapshot_sha256"]),
    }
    if source_cache:
        values["PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE"] = (
            "/worker-state/source-identity-cache.json"
        )
    return tuple(sorted(values.items()))


def _docker_argv(
    manifest: Mapping[str, object], host: Mapping[str, object],
    *, module: str, module_argv: Sequence[str],
) -> tuple[str, ...]:
    roots = host["roots"]
    expected = host["expected"]
    producer = manifest["producer"]
    assert isinstance(roots, Mapping)
    assert isinstance(expected, Mapping)
    assert isinstance(producer, Mapping)
    mounts = (
        ("snapshot_root", "/pq", True),
        ("model_root", "/model", True),
        ("dataset_path", "/dataset", True),
        ("run_root", "/run", False),
        ("worker_state_root", "/worker-state", False),
    )
    argv: list[str] = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "--uts=host",
        "--user", f"{expected['uid']}:{expected['gid']}",
        "--workdir", "/worker-state/tmp",
    ]
    for field, destination, readonly in mounts:
        source = _safe_mount_source(roots[field], field=field)
        spec = f"type=bind,src={source},dst={destination}"
        if readonly:
            spec += ",readonly"
        argv.extend(("--mount", spec))
    for name, value in _container_environment(manifest):
        argv.extend(("--env", f"{name}={value}"))
    image_ref = f"{IMAGE_REPOSITORY}@{producer['image_digest']}"
    argv.extend((
        "--entrypoint", "/bin/bash", image_ref, "-c", _BOOTSTRAP_SCRIPT,
        "prismaquant-campaign-bootstrap", module, *map(str, module_argv),
    ))
    return tuple(argv)


def _preflight_argv(
    manifest: Mapping[str, object], host: Mapping[str, object],
) -> tuple[str, ...]:
    roots = host["roots"]
    expected = host["expected"]
    producer = manifest["producer"]
    assert isinstance(roots, Mapping)
    assert isinstance(expected, Mapping)
    assert isinstance(producer, Mapping)
    snapshot = _safe_mount_source(roots["snapshot_root"], field="snapshot_root")
    gpu = expected["gpu"]
    assert isinstance(gpu, Mapping)
    return (
        "python3", "-P", "-B", "-s",
        f"{snapshot}/tools/prismaquant_source_bootstrap.py",
        "run-module", "--source-root", snapshot,
        "prismaquant.rtx4090_two_host_campaign", "worker-preflight",
        "--host-id", str(host["id"]),
        "--hostname", str(expected["hostname"]),
        "--uid", str(expected["uid"]), "--gid", str(expected["gid"]),
        "--gpu-name", str(gpu["name"]),
        "--gpu-uuid", str(gpu["uuid"]),
        "--gpu-cc", ".".join(str(value) for value in gpu["compute_capability"]),
        "--gpu-count", str(gpu["device_count"]),
        "--image-ref", f"{IMAGE_REPOSITORY}@{producer['image_digest']}",
        "--commit", str(producer["commit"]),
        "--tree", str(producer["tree"]),
        "--closure-sha256", str(producer["snapshot_sha256"]),
        "--model-root", str(roots["model_root"]),
        "--dataset-path", str(roots["dataset_path"]),
        "--snapshot-root", snapshot,
        "--run-root", str(roots["run_root"]),
        "--worker-state-root", str(roots["worker_state_root"]),
    )


def _stage_module_argv(
    stage: str, index: int,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    worker = f"/run/worker-{index}"
    if stage == "prepare_calibration":
        return "prismaquant.sample_parallel_probe", (
            "prepare-calibration", "--model", "/model", "--dataset", "/dataset",
            "--nsamples", "32", "--seqlen", "1024", "--calib-seed", "42",
            "--partitions", "2", "--output", "/run/calibration.pt",
            "--manifest-output", "/run/calibration.json",
        ), ("/model", "/dataset"), ("/run/calibration.pt", "/run/calibration.json")
    if stage == "coordinator_source_identity":
        return "prismaquant.sample_parallel_probe", (
            "prepare-worker-source-cache", "--model", "/model",
            "--output", "/run/coordinator/source-identity-cache.json",
            "--offload-folder", "/run/coordinator/source-identity-offload",
        ), ("/model",), ("/run/coordinator/source-identity-cache.json",)
    if stage == "prepare_run_contract":
        return "prismaquant.sample_parallel_probe", (
            "prepare-run-contract", "--model", "/model", "--dataset", "/dataset",
            "--calib-seed", "42", "--dtype", "bf16", "--importance-weighting",
            "--emit-marginals", "--activation-rows-limit", "1024",
            "--producer-snapshot-root", "/pq", "--container-image-digest",
            "__IMAGE_DIGEST__", "--source-identity-cache",
            "/run/coordinator/source-identity-cache.json", "--output",
            "/run/run-contract.json",
        ), ("/run/coordinator/source-identity-cache.json",), ("/run/run-contract.json",)
    if stage == "build_sample_cover":
        return "prismaquant.sample_parallel_probe", (
            "build-cover", "--calibration-manifest", "/run/calibration.json",
            "--run-contract", "/run/run-contract.json", "--output", "/run/cover.json",
        ), ("/run/calibration.json", "/run/run-contract.json"), ("/run/cover.json",)
    if stage == "worker_source_identity":
        return "prismaquant.sample_parallel_probe", (
            "prepare-worker-source-cache", "--model", "/model", "--output",
            "/worker-state/source-identity-cache.json", "--offload-folder",
            "/worker-state/source-identity-offload",
        ), ("/model",), ("/worker-state/source-identity-cache.json",)
    common_probe = (
        "--model", "/model", "--dataset", "/dataset", "--seqlen", "1024",
        "--calib-seed", "42", "--dtype", "bf16",
        "--global-calibration-tensor", "/run/calibration.pt",
        "--sample-partition-index", str(index), "--sample-run-contract",
        "/run/run-contract.json", "--sample-cover", "/run/cover.json",
        "--producer-snapshot-root", "/pq", "--container-image-digest",
        "__IMAGE_DIGEST__", "--importance-weighting", "--emit-marginals",
        "--include-mtp", "--include-lm-head", "--no-include-visual",
        "--unified-sweep", "--activation-rows-limit", "1024",
        "--activation-cache-dir", f"{worker}/act", "--work-dir",
        f"{worker}/work", "--output", f"{worker}/probe.pkl",
    )
    if stage == "sample_ce":
        return "prismaquant.incremental_probe", (
            *common_probe, "--sample-importance-stats-output", f"{worker}/ce.json",
        ), ("/run/calibration.pt", "/run/run-contract.json", "/run/cover.json"), (
            f"{worker}/ce.json",
        )
    if stage == "merge_importance":
        return "prismaquant.sample_parallel_probe", (
            "merge-importance", "--local-stats", "/run/worker-0/ce.json",
            "/run/worker-1/ce.json", "--output", "/run/global-ce.json",
        ), ("/run/worker-0/ce.json", "/run/worker-1/ce.json"), ("/run/global-ce.json",)
    if stage == "sample_fisher":
        return "prismaquant.incremental_probe", (
            *common_probe, "--sample-global-importance-receipt", "/run/global-ce.json",
        ), ("/run/global-ce.json",), (f"{worker}/probe.pkl", f"{worker}/act")
    if stage == "merge_sample_probe":
        return "prismaquant.sample_parallel_probe", (
            "merge", "--cover", "/run/cover.json", "--probe-shards",
            "/run/worker-0/probe.pkl", "/run/worker-1/probe.pkl",
            "--activation-cache", "0=/run/worker-0/act", "1=/run/worker-1/act",
            "--output-bundle", "/run/merged", "--max-rows", "1024",
        ), ("/run/cover.json", "/run/worker-0/probe.pkl", "/run/worker-1/probe.pkl"), (
            "/run/merged/probe.pkl", "/run/merged/activation_cache", "/run/merged/commit.json",
        )
    if stage == "derive_col_weights":
        return "prismaquant.rtx4090_fp8_burn", (
            "derive-col-weights", "--sample-merge-bundle", "/run/merged",
            "--output", "/run/cb_col_weights.pkl",
        ), ("/run/merged",), ("/run/cb_col_weights.pkl",)
    if stage == "attest_execution":
        return "prismaquant.rtx4090_fp8_burn", (
            "attest-execution", "--sample-run-contract", "/run/run-contract.json",
            "--producer-snapshot", SNAPSHOT_MANIFEST, "--launcher-image-digest",
            "__IMAGE_DIGEST__", "--output", "/run/burn/execution-attestation.json",
        ), ("/run/run-contract.json", SNAPSHOT_MANIFEST), (
            "/run/burn/execution-attestation.json",
        )
    burn_common = (
        "--model", "/model", "--probe", "/run/merged/probe.pkl",
        "--col-weights", "/run/cb_col_weights.pkl", "--dataset", "/dataset",
    )
    if stage == "prepare_burn":
        return "prismaquant.rtx4090_fp8_burn", (
            "prepare", *burn_common, "--source-identity",
            "/run/coordinator/source-identity-cache.json", "--producer-snapshot",
            SNAPSHOT_MANIFEST, "--execution-attestation",
            "/run/burn/execution-attestation.json", "--launcher-image-digest",
            "__IMAGE_DIGEST__", "--sample-merge-commit", "/run/merged/commit.json",
            "--activation-cache-dir", "/run/merged/activation_cache", "--output-dir",
            "/run/burn/plan", "--n-calib-samples", "32", "--calib-seqlen", "1024",
            "--calib-seed", "42",
        ), ("/run/merged", "/run/cb_col_weights.pkl"), (
            "/run/burn/plan/campaign-plan.json", "/run/burn/plan/stripe-00.qnames.txt",
            "/run/burn/plan/stripe-01.qnames.txt",
        )
    if stage == "measure_burn":
        return "prismaquant.rtx4090_fp8_burn", (
            "measure", *burn_common, "--source-identity",
            "/worker-state/source-identity-cache.json", "--producer-snapshot",
            SNAPSHOT_MANIFEST, "--execution-attestation",
            "/run/burn/execution-attestation.json", "--launcher-image-digest",
            "__IMAGE_DIGEST__", "--sample-merge-commit", "/run/merged/commit.json",
            "--activation-cache-dir", "/run/merged/activation_cache", "--plan",
            "/run/burn/plan/campaign-plan.json", "--stripe", str(index),
            "--checkpoint-dir", f"/worker-state/burn-stripe-{index}", "--output",
            f"/run/burn/stripe-{index}.pkl",
        ), ("/run/burn/plan/campaign-plan.json", "/run/merged"), (
            f"/run/burn/stripe-{index}.pkl",
        )
    if stage == "merge_burn":
        return "prismaquant.rtx4090_fp8_burn", (
            "merge", "--plan", "/run/burn/plan/campaign-plan.json",
            "--producer-snapshot", SNAPSHOT_MANIFEST, "--col-weights",
            "/run/cb_col_weights.pkl", "--shards", "/run/burn/stripe-0.pkl",
            "/run/burn/stripe-1.pkl", "--output", "/run/burn/aura-merged.pkl",
        ), ("/run/burn/stripe-0.pkl", "/run/burn/stripe-1.pkl"), (
            "/run/burn/aura-merged.pkl",
        )
    if stage == "allocate":
        return "prismaquant.rtx4090_fp8_burn", (
            "allocate", "--plan", "/run/burn/plan/campaign-plan.json",
            "--producer-snapshot", SNAPSHOT_MANIFEST, "--model", "/model",
            "--probe", "/run/merged/probe.pkl", "--sample-merge-commit",
            "/run/merged/commit.json", "--activation-cache-dir",
            "/run/merged/activation_cache", "--col-weights",
            "/run/cb_col_weights.pkl", "--merged", "/run/burn/aura-merged.pkl",
            "--cost-output", "/run/burn/allocator-cost.pkl", "--output-dir",
            "/run/burn/allocation", "--threads", "16",
        ), ("/run/burn/aura-merged.pkl", "/run/merged"), (
            "/run/burn/allocator-cost.pkl", "/run/burn/allocation/layer_config.json",
        )
    raise RTX4090TwoHostCampaignError(f"unsupported fixed campaign stage {stage!r}")


def build_stage_command(
    manifest: Mapping[str, object], assignment: StageAssignment,
) -> CampaignCommand:
    normalized = validate_campaign_manifest(manifest)
    host = _host_by_id(normalized, assignment.host_id)
    worker_index = _worker_index(normalized, assignment.host_id)
    producer = normalized["producer"]
    assert isinstance(producer, Mapping)
    if assignment.stage == "host_preflight":
        argv = _preflight_argv(normalized, host)
        return CampaignCommand(
            work_id=assignment.work_id, stage=assignment.stage,
            host_id=assignment.host_id, assignment_index=assignment.assignment_index,
            worker_index=worker_index,
            module=None, module_argv=(), argv=argv, resume_argv=None,
            env=_HOST_ENV,
            inputs=(), outputs=(), gpu_bearing=False,
        )
    module, raw_module_argv, inputs, outputs = _stage_module_argv(
        assignment.stage, worker_index,
    )
    module_argv = tuple(
        str(producer["image_digest"]) if value == "__IMAGE_DIGEST__" else value
        for value in raw_module_argv
    )
    argv = _docker_argv(normalized, host, module=module, module_argv=module_argv)
    resume_argv = None
    if assignment.stage in _RESUMABLE_FLAG_STAGES:
        resume_module_argv = (*module_argv, "--resume")
        resume_argv = _docker_argv(
            normalized, host, module=module, module_argv=resume_module_argv,
        )
    return CampaignCommand(
        work_id=assignment.work_id, stage=assignment.stage,
        host_id=assignment.host_id, assignment_index=assignment.assignment_index,
        worker_index=worker_index,
        module=module, module_argv=module_argv, argv=argv,
        resume_argv=resume_argv, env=_HOST_ENV, inputs=inputs, outputs=outputs,
        gpu_bearing=assignment.stage in _GPU_STAGES,
    )


def _all_assignments(
    manifest: Mapping[str, object],
) -> tuple[StageAssignment, ...]:
    state = initial_campaign_state(manifest)
    result: list[StageAssignment] = []
    while True:
        ready = next_ready_assignments(manifest, state)
        if not ready:
            break
        for assignment in ready:
            result.append(assignment)
            state = complete_assignment(
                manifest, state, assignment, "0" * 64,
            )
    if len({assignment.work_id for assignment in result}) != len(result):
        raise RTX4090TwoHostCampaignError("campaign DAG produced duplicate work ids")
    return tuple(result)


def build_command_plan(manifest: Mapping[str, object]) -> dict[str, object]:
    normalized = validate_campaign_manifest(manifest)
    commands = [
        build_stage_command(normalized, assignment).to_dict()
        for assignment in _all_assignments(normalized)
    ]
    body: dict[str, object] = {
        "schema": COMMAND_PLAN_SCHEMA,
        "campaign_identity_sha256": normalized["identity_sha256"],
        "canonical_container_paths": dict(CANONICAL_CONTAINER_PATHS),
        "image_repository": IMAGE_REPOSITORY,
        "exactly_two_hosts": True,
        "partition_equals_stripe_assignment": True,
        "commands": commands,
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def validate_command_plan(
    raw: Mapping[str, object], manifest: Mapping[str, object],
) -> dict[str, object]:
    expected = build_command_plan(manifest)
    if dict(raw) != expected:
        raise RTX4090TwoHostCampaignError(
            "command plan differs from the fixed campaign DAG/arguments"
        )
    return expected


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"


def _write_exact_no_clobber(path: Path, value: Mapping[str, object]) -> None:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise RTX4090TwoHostCampaignError(
            f"existing campaign artifact differs: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if not path.is_file() or path.read_bytes() != payload:
                raise RTX4090TwoHostCampaignError(
                    f"racing campaign artifact differs: {path}"
                ) from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_state(path: Path, state: Mapping[str, object]) -> None:
    payload = _json_bytes(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _plain(value: object) -> object:
    serializer = getattr(value, "to_payload", None)
    if callable(serializer):
        return _plain(serializer())
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, bytes):
        return {
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise RTX4090TwoHostCampaignError(
        f"transport receipt contains non-canonical value {type(value).__name__}"
    )


def _normalize_job_receipt(value: object, request: RunRequest) -> dict[str, object]:
    plain = _plain(value)
    try:
        receipt = JobReceipt.from_payload(plain)
    except (TypeError, ValueError) as exc:
        raise RTX4090TwoHostCampaignError(
            "transport returned a malformed job receipt"
        ) from exc
    result = receipt.to_payload()
    if result.get("job_id") != request.job_id:
        raise RTX4090TwoHostCampaignError("transport receipt job id differs")
    if result.get("request_sha256") != request.request_sha256:
        raise RTX4090TwoHostCampaignError("transport receipt request digest differs")
    if result.get("state") != "succeeded" or result.get("returncode") != 0:
        raise RTX4090TwoHostCampaignError(
            f"transport job {request.job_id} did not succeed"
        )
    return result


def _normalize_telemetry(
    values: Sequence[object], *, required: bool,
    expected_gpu: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    result: list[dict[str, object]] = []
    normalized_snapshots: list[TelemetrySnapshot] = []
    for index, value in enumerate(values):
        plain = _plain(value)
        try:
            snapshot = TelemetrySnapshot.from_payload(plain)
        except (TypeError, ValueError) as exc:
            raise RTX4090TwoHostCampaignError(
                f"telemetry sample {index} is malformed"
            ) from exc
        normalized_snapshots.append(snapshot)
        result.append(snapshot.to_payload())
    if required and not result:
        raise RTX4090TwoHostCampaignError(
            "GPU-bearing assignment returned no utilization telemetry"
        )
    expected_count = int(expected_gpu["device_count"])
    expected_name = str(expected_gpu["name"])
    expected_uuid = str(expected_gpu["uuid"])
    previous_ns = 0
    active_samples = 0
    utilizations: list[float] = []
    for index, snapshot in enumerate(result):
        captured_ns = int(snapshot["captured_ns"])
        if captured_ns <= previous_ns:
            raise RTX4090TwoHostCampaignError(
                "telemetry captured_ns values are not strictly increasing"
            )
        previous_ns = captured_ns
        gpus = snapshot["gpus"]
        if not isinstance(gpus, list) or len(gpus) != expected_count:
            raise RTX4090TwoHostCampaignError(
                f"telemetry sample {index} has the wrong visible GPU count"
            )
        matches = [
            gpu for gpu in gpus
            if isinstance(gpu, Mapping)
            and gpu.get("name") == expected_name
            and gpu.get("uuid") == expected_uuid
        ]
        if len(matches) != 1:
            raise RTX4090TwoHostCampaignError(
                f"telemetry sample {index} differs from manifest GPU identity"
            )
        utilization = matches[0].get("gpu_utilization_pct")
        if utilization is not None:
            value = float(utilization)
            utilizations.append(value)
            active_samples += int(value > 0.0)
    if required and not any(value > 0.0 for value in utilizations):
        raise RTX4090TwoHostCampaignError(
            "GPU-bearing assignment has no positive utilization sample"
        )
    shared_summary = summarize_utilization(normalized_snapshots).to_payload()
    summary: dict[str, object] = {
        **shared_summary,
        "first_captured_ns": result[0]["captured_ns"] if result else None,
        "last_captured_ns": result[-1]["captured_ns"] if result else None,
        "coverage_ns": (
            int(result[-1]["captured_ns"]) - int(result[0]["captured_ns"])
            if result else 0
        ),
        "live_cuda_observed": active_samples > 0,
    }
    return result, summary


def _stage_receipt(
    manifest: Mapping[str, object], command: CampaignCommand,
    request: RunRequest, job_receipt: object, telemetry: Sequence[object],
    dependency_receipt_sha256s: Sequence[str],
) -> dict[str, object]:
    job = _normalize_job_receipt(job_receipt, request)
    host = _host_by_id(manifest, command.host_id)
    expected = host["expected"]
    assert isinstance(expected, Mapping)
    expected_gpu = expected["gpu"]
    assert isinstance(expected_gpu, Mapping)
    samples, telemetry_summary = _normalize_telemetry(
        telemetry, required=command.gpu_bearing, expected_gpu=expected_gpu,
    )
    body: dict[str, object] = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "campaign_identity_sha256": manifest["identity_sha256"],
        "host_identity_sha256": canonical_sha256(host),
        "producer_identity_sha256": canonical_sha256(manifest["producer"]),
        "work_id": command.work_id,
        "stage": command.stage,
        "host_id": command.host_id,
        "assignment_index": command.assignment_index,
        "command_identity_sha256": command.identity_sha256,
        "request_sha256": request.request_sha256,
        "dependency_receipt_sha256s": list(dependency_receipt_sha256s),
        "job": job,
        "gpu_bearing": command.gpu_bearing,
        "telemetry": samples,
        "telemetry_sha256": canonical_sha256(samples),
        "telemetry_summary": telemetry_summary,
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


class CampaignCoordinator:
    def __init__(
        self,
        manifest: Mapping[str, object],
        state_dir: str | Path,
        *,
        transports: Mapping[str, CampaignTransport] | None = None,
    ):
        self.manifest = validate_campaign_manifest(manifest)
        self.state_dir = Path(state_dir)
        if not self.state_dir.is_absolute() or self.state_dir == Path("/"):
            raise RTX4090TwoHostCampaignError(
                "campaign state directory must be a non-root absolute path"
            )
        if self.state_dir.is_symlink():
            raise RTX4090TwoHostCampaignError("campaign state directory is a symlink")
        self.transports = dict(transports or {})
        self.commands = {
            command.work_id: command
            for command in (
                build_stage_command(self.manifest, assignment)
                for assignment in _all_assignments(self.manifest)
            )
        }

    @property
    def state_path(self) -> Path:
        return self.state_dir / STATE_FILE

    @contextmanager
    def _exclusive_lock(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_dir / LOCK_FILE
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise RTX4090TwoHostCampaignError(
                "cannot open the campaign coordinator lock"
            ) from exc
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RTX4090TwoHostCampaignError(
                    "another campaign coordinator holds the state lock"
                ) from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def initialize(self) -> dict[str, object]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        _write_exact_no_clobber(
            self.state_dir / MANIFEST_FILE, self.manifest,
        )
        _write_exact_no_clobber(
            self.state_dir / PLAN_FILE, build_command_plan(self.manifest),
        )
        if self.state_path.exists():
            raise RTX4090TwoHostCampaignError(
                "campaign state already exists; use resume"
            )
        state = initial_campaign_state(self.manifest)
        _replace_state(self.state_path, state)
        return state

    def load_state(self) -> dict[str, object]:
        if not self.state_path.is_file():
            raise RTX4090TwoHostCampaignError("campaign state is absent; use run")
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RTX4090TwoHostCampaignError("campaign state is unreadable") from exc
        if not isinstance(raw, Mapping):
            raise RTX4090TwoHostCampaignError("campaign state is not an object")
        return validate_campaign_state(raw, self.manifest)

    def _request(self, command: CampaignCommand, *, resume: bool) -> RunRequest:
        argv = (
            command.resume_argv
            if resume and command.resume_argv is not None
            else command.argv
        )
        campaign_prefix = str(self.manifest["identity_sha256"])[:16]
        job_id = (
            f"pq-{campaign_prefix}-{command.assignment_index:02d}-"
            f"{command.stage}-{command.host_id}"
        )
        return RunRequest(
            job_id=job_id,
            argv=argv,
            cwd="/",
            env=command.env,
            timeout_seconds=None,
            inherit_env=False,
        )

    def _execute(
        self, assignment: StageAssignment, *, resume: bool,
        dependency_receipt_sha256s: Sequence[str],
    ) -> tuple[StageAssignment, dict[str, object]]:
        command = self.commands[assignment.work_id]
        transport = self.transports.get(assignment.host_id)
        if transport is None:
            raise RTX4090TwoHostCampaignError(
                f"no telemetry-capable transport for host {assignment.host_id!r}"
            )
        runner = getattr(transport, "run_with_telemetry", None)
        if not callable(runner):
            raise RTX4090TwoHostCampaignError(
                "live transport lacks run_with_telemetry; refusing an "
                "unmeasured GPU campaign"
            )
        request = self._request(command, resume=resume)
        job, samples = runner(request)
        return assignment, _stage_receipt(
            self.manifest, command, request, job, samples,
            dependency_receipt_sha256s,
        )

    @staticmethod
    def _dependency_receipts(
        state: Mapping[str, object], stage: str,
    ) -> tuple[str, ...]:
        dependencies = _STAGE_DEPENDENCIES[stage]
        rows = state["completions"]
        assert isinstance(rows, list)
        return tuple(sorted(
            str(row["receipt_sha256"])
            for row in rows
            if isinstance(row, Mapping) and row.get("stage") in dependencies
        ))

    def _validate_receipt(
        self,
        raw: object,
        assignment: StageAssignment,
        state: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(raw, Mapping) or set(raw) != _EXECUTION_RECEIPT_KEYS:
            raise RTX4090TwoHostCampaignError(
                f"execution receipt fields differ for {assignment.work_id}"
            )
        receipt = dict(raw)
        body = dict(receipt)
        digest = body.pop("identity_sha256", None)
        if digest != canonical_sha256(body):
            raise RTX4090TwoHostCampaignError(
                f"execution receipt digest differs for {assignment.work_id}"
            )
        command = self.commands[assignment.work_id]
        host = _host_by_id(self.manifest, assignment.host_id)
        exact = {
            "schema": EXECUTION_RECEIPT_SCHEMA,
            "campaign_identity_sha256": self.manifest["identity_sha256"],
            "host_identity_sha256": canonical_sha256(host),
            "producer_identity_sha256": canonical_sha256(
                self.manifest["producer"]
            ),
            "work_id": assignment.work_id,
            "stage": assignment.stage,
            "host_id": assignment.host_id,
            "assignment_index": assignment.assignment_index,
            "command_identity_sha256": command.identity_sha256,
            "gpu_bearing": command.gpu_bearing,
            "dependency_receipt_sha256s": list(
                self._dependency_receipts(state, assignment.stage)
            ),
        }
        for key, expected in exact.items():
            if receipt.get(key) != expected:
                raise RTX4090TwoHostCampaignError(
                    f"execution receipt {key} differs for {assignment.work_id}"
                )
        request_candidates = (
            self._request(command, resume=False),
            self._request(command, resume=True),
        )
        request = next((
            item for item in request_candidates
            if item.request_sha256 == receipt.get("request_sha256")
        ), None)
        if request is None:
            raise RTX4090TwoHostCampaignError(
                f"execution receipt request differs for {assignment.work_id}"
            )
        normalized_job = _normalize_job_receipt(receipt["job"], request)
        if receipt["job"] != normalized_job:
            raise RTX4090TwoHostCampaignError(
                f"execution job receipt is noncanonical for {assignment.work_id}"
            )
        expected = host["expected"]
        assert isinstance(expected, Mapping)
        expected_gpu = expected["gpu"]
        assert isinstance(expected_gpu, Mapping)
        telemetry = receipt["telemetry"]
        if not isinstance(telemetry, list):
            raise RTX4090TwoHostCampaignError(
                f"execution telemetry is malformed for {assignment.work_id}"
            )
        samples, summary = _normalize_telemetry(
            telemetry,
            required=command.gpu_bearing,
            expected_gpu=expected_gpu,
        )
        if (
            telemetry != samples
            or receipt.get("telemetry_sha256") != canonical_sha256(samples)
            or receipt.get("telemetry_summary") != summary
        ):
            raise RTX4090TwoHostCampaignError(
                f"execution telemetry differs for {assignment.work_id}"
            )
        return receipt

    def _read_receipt(
        self, assignment: StageAssignment, state: Mapping[str, object],
    ) -> dict[str, object]:
        path = self.state_dir / RECEIPT_DIR / f"{assignment.work_id}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RTX4090TwoHostCampaignError(
                f"execution receipt is unreadable: {path}"
            ) from exc
        return self._validate_receipt(raw, assignment, state)

    def advance(
        self, state: Mapping[str, object], *, resume: bool,
    ) -> dict[str, object]:
        ready = next_ready_assignments(self.manifest, state)
        if not ready:
            return dict(state)
        current = dict(state)
        pending: list[StageAssignment] = []
        for assignment in ready:
            receipt_path = (
                self.state_dir / RECEIPT_DIR / f"{assignment.work_id}.json"
            )
            if not receipt_path.exists():
                pending.append(assignment)
                continue
            receipt = self._read_receipt(assignment, current)
            current = complete_assignment(
                self.manifest,
                current,
                assignment,
                str(receipt["identity_sha256"]),
            )
            _replace_state(self.state_path, current)
        if not pending:
            return current
        results: list[tuple[StageAssignment, dict[str, object]]] = []
        failures: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futures = {
                pool.submit(
                    self._execute,
                    assignment,
                    resume=resume,
                    dependency_receipt_sha256s=self._dependency_receipts(
                        current, assignment.stage,
                    ),
                ): assignment
                for assignment in pending
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except BaseException as exc:  # retain successful peer progress
                    failures.append(exc)
        for assignment, receipt in sorted(
            results, key=lambda item: item[0].assignment_index,
        ):
            receipt_path = self.state_dir / RECEIPT_DIR / f"{assignment.work_id}.json"
            _write_exact_no_clobber(receipt_path, receipt)
            current = complete_assignment(
                self.manifest, current, assignment,
                str(receipt["identity_sha256"]),
            )
            _replace_state(self.state_path, current)
        if failures:
            raise RTX4090TwoHostCampaignError(
                f"campaign barrier had {len(failures)} failed assignment(s): "
                f"{failures[0]}"
            ) from failures[0]
        return current

    def run_to_completion(self, *, resume: bool) -> dict[str, object]:
        with self._exclusive_lock():
            state = self.load_state() if resume else self.initialize()
            while next_ready_assignments(self.manifest, state):
                state = self.advance(state, resume=resume)
            return self.verify()

    def status(self) -> dict[str, object]:
        state = self.load_state()
        ready = next_ready_assignments(self.manifest, state)
        completions = state["completions"]
        assert isinstance(completions, list)
        return {
            "schema": STATUS_SCHEMA,
            "campaign_identity_sha256": self.manifest["identity_sha256"],
            "completed_assignments": len(completions),
            "total_assignments": len(self.commands),
            "complete": len(completions) == len(self.commands) and not ready,
            "ready": [assignment.work_id for assignment in ready],
            "state_identity_sha256": state["identity_sha256"],
        }

    def verify(self) -> dict[str, object]:
        manifest_path = self.state_dir / MANIFEST_FILE
        plan_path = self.state_dir / PLAN_FILE
        try:
            manifest_bytes = manifest_path.read_bytes()
            plan_bytes = plan_path.read_bytes()
            stored_manifest = parse_campaign_manifest(
                manifest_bytes.decode("utf-8")
            )
            raw_plan = json.loads(plan_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RTX4090TwoHostCampaignError(
                "stored campaign manifest/plan is unreadable"
            ) from exc
        if stored_manifest != self.manifest:
            raise RTX4090TwoHostCampaignError("stored campaign manifest differs")
        if manifest_bytes != _json_bytes(self.manifest):
            raise RTX4090TwoHostCampaignError(
                "stored campaign manifest bytes are noncanonical"
            )
        if not isinstance(raw_plan, Mapping):
            raise RTX4090TwoHostCampaignError("stored command plan is malformed")
        expected_plan = validate_command_plan(raw_plan, self.manifest)
        if plan_bytes != _json_bytes(expected_plan):
            raise RTX4090TwoHostCampaignError(
                "stored command plan bytes are noncanonical"
            )
        state = self.load_state()
        completions = state["completions"]
        assert isinstance(completions, list)
        completion_by_work_id = {
            str(row["work_id"]): row
            for row in completions
            if isinstance(row, Mapping)
        }
        assignment_by_work_id = {
            assignment.work_id: assignment
            for assignment in _all_assignments(self.manifest)
        }
        receipt_root = self.state_dir / RECEIPT_DIR
        paths = sorted(receipt_root.glob("*.json")) if receipt_root.exists() else []
        if {path.stem for path in paths} != set(completion_by_work_id):
            raise RTX4090TwoHostCampaignError(
                "execution receipt files differ from durable state completions"
            )
        for path in paths:
            work_id = path.stem
            assignment = assignment_by_work_id.get(work_id)
            if assignment is None:
                raise RTX4090TwoHostCampaignError(
                    f"execution receipt is outside the fixed DAG: {path}"
                )
            receipt = self._read_receipt(assignment, state)
            completion = completion_by_work_id[work_id]
            if completion.get("receipt_sha256") != receipt["identity_sha256"]:
                raise RTX4090TwoHostCampaignError(
                    f"execution receipt is not state-bound: {path}"
                )
            if path.read_bytes() != _json_bytes(receipt):
                raise RTX4090TwoHostCampaignError(
                    f"execution receipt bytes are noncanonical: {path}"
                )
        return self.status()


def _load_manifest(path: str | Path) -> dict[str, object]:
    try:
        return parse_campaign_manifest(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RTX4090TwoHostCampaignError(
            f"campaign manifest is invalid: {path}"
        ) from exc


def _worker_preflight(args: argparse.Namespace) -> dict[str, object]:
    if os.getuid() != args.uid or os.getgid() != args.gid:
        raise RTX4090TwoHostCampaignError("host UID/GID differs from manifest")
    if socket.gethostname() != args.hostname:
        raise RTX4090TwoHostCampaignError("host name differs from manifest")
    model = Path(args.model_root)
    dataset = Path(args.dataset_path)
    snapshot = Path(args.snapshot_root)
    run_root = Path(args.run_root)
    worker_root = Path(args.worker_state_root)
    if not model.is_dir() or model.is_symlink():
        raise RTX4090TwoHostCampaignError("model root is absent or unsafe")
    if not dataset.is_file() or dataset.is_symlink():
        raise RTX4090TwoHostCampaignError("dataset path is absent or unsafe")
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise RTX4090TwoHostCampaignError("snapshot root is absent or unsafe")
    for root in (run_root, worker_root):
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not os.access(root, os.W_OK | os.X_OK):
            raise RTX4090TwoHostCampaignError("writable campaign root is unsafe")
    snapshot_tool = snapshot / "tools/prismaquant_runtime_snapshot.py"
    verified = subprocess.run((
        sys.executable, "-P", "-B", "-s", str(snapshot_tool), "verify",
        "--snapshot", str(snapshot), "--expected-commit", args.commit,
        "--expected-tree", args.tree, "--expected-closure-sha256",
        args.closure_sha256,
    ), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if verified.returncode != 0:
        raise RTX4090TwoHostCampaignError("runtime snapshot verification failed")
    repo_digests = subprocess.run(
        ("docker", "image", "inspect", "--format",
         "{{range .RepoDigests}}{{println .}}{{end}}", args.image_ref),
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if repo_digests.returncode != 0 or args.image_ref not in {
        line.strip() for line in repo_digests.stdout.splitlines()
    }:
        raise RTX4090TwoHostCampaignError("Docker RepoDigest differs")
    gpu_query = subprocess.run((
        "nvidia-smi", "--query-gpu=name,uuid,compute_cap",
        "--format=csv,noheader,nounits",
    ), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rows = [line.strip() for line in gpu_query.stdout.splitlines() if line.strip()]
    if gpu_query.returncode != 0 or len(rows) != args.gpu_count:
        raise RTX4090TwoHostCampaignError("visible GPU count differs")
    fields = [field.strip() for field in rows[0].split(",")]
    if fields != [args.gpu_name, args.gpu_uuid, args.gpu_cc]:
        raise RTX4090TwoHostCampaignError("visible GPU identity differs")
    return {
        "host_id": args.host_id,
        "hostname": args.hostname,
        "uid": args.uid,
        "gid": args.gid,
        "gpu": {"name": fields[0], "uuid": fields[1], "compute_capability": fields[2]},
        "image_ref": args.image_ref,
        "snapshot_closure_sha256": args.closure_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--output", required=True)
    for name in ("run", "resume", "status", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--state-dir", required=True)
    worker = sub.add_parser("worker-preflight", help=argparse.SUPPRESS)
    for flag in (
        "host-id", "hostname", "gpu-name", "gpu-uuid", "gpu-cc", "image-ref",
        "commit", "tree", "closure-sha256", "model-root", "dataset-path",
        "snapshot-root", "run-root", "worker-state-root",
    ):
        worker.add_argument(f"--{flag}", required=True)
    worker.add_argument("--uid", required=True, type=int)
    worker.add_argument("--gid", required=True, type=int)
    worker.add_argument("--gpu-count", required=True, type=int)
    return parser


TransportFactory = Callable[[Mapping[str, object]], Mapping[str, CampaignTransport]]


def main(
    argv: Sequence[str] | None = None,
    *,
    transport_factory: TransportFactory | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if args.command == "worker-preflight":
        print(json.dumps(_worker_preflight(args), sort_keys=True))
        return 0
    manifest = _load_manifest(args.manifest)
    if args.command == "plan":
        _write_exact_no_clobber(Path(args.output), build_command_plan(manifest))
        print(args.output)
        return 0
    transports = transport_factory(manifest) if transport_factory is not None else {}
    coordinator = CampaignCoordinator(
        manifest, args.state_dir, transports=transports,
    )
    if args.command == "run":
        result = coordinator.run_to_completion(resume=False)
    elif args.command == "resume":
        result = coordinator.run_to_completion(resume=True)
    elif args.command == "status":
        result = coordinator.status()
    else:
        result = coordinator.verify()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RTX4090TwoHostCampaignError as exc:
        print(f"rtx4090-two-host-campaign: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
