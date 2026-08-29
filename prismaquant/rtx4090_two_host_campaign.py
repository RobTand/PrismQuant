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
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence

from prismaquant.cluster_campaign_contract import (
    CAMPAIGN_MANIFEST_SCHEMA,
    CANONICAL_CONTAINER_PATHS,
    LEGACY_CAMPAIGN_MANIFEST_SCHEMA,
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
    ClusterTransportError,
    JobConflictError,
    JobNotFoundError,
    JobReceipt,
    RunRequest,
    TelemetryUnavailableError,
    TelemetrySnapshot,
    TreeManifest,
    build_tree_manifest,
    canonical_json_bytes,
    list_regular_file_children_nofollow,
    read_regular_file_nofollow,
    summarize_utilization,
)


COMMAND_PLAN_SCHEMA = "prismaquant.rtx4090_two_host_campaign.command_plan.v2"
COMMAND_SCHEMA = "prismaquant.rtx4090_two_host_campaign.command.v1"
LEGACY_EXECUTION_RECEIPT_SCHEMA = (
    "prismaquant.rtx4090_two_host_campaign.execution_receipt.v1"
)
EXECUTION_RECEIPT_SCHEMA = (
    "prismaquant.rtx4090_two_host_campaign.execution_receipt.v2"
)
STATUS_SCHEMA = "prismaquant.rtx4090_two_host_campaign.status.v1"
IMAGE_REPOSITORY = "eugr/spark-vllm"
SOURCE_BOOTSTRAP = "/pq/tools/prismaquant_source_bootstrap.py"
SNAPSHOT_MANIFEST = "/pq/.prismaquant-runtime-snapshot.json"
STATE_FILE = "campaign-state.json"
PLAN_FILE = "command-plan.json"
MANIFEST_FILE = "campaign-manifest.json"
RECEIPT_DIR = "receipts"
ATTEMPT_TELEMETRY_DIR = "attempt-telemetry"
LOCK_FILE = "campaign.lock"
LEGACY_ATTEMPT_TELEMETRY_SCHEMA = (
    "prismaquant.rtx4090_two_host_campaign.attempt_telemetry.v1"
)
LEGACY_ATTEMPT_TELEMETRY_V2_SCHEMA = (
    "prismaquant.rtx4090_two_host_campaign.attempt_telemetry.v2"
)
ATTEMPT_TELEMETRY_SCHEMA = (
    "prismaquant.rtx4090_two_host_campaign.attempt_telemetry.v3"
)
ATTEMPT_TELEMETRY_RECORD_SCHEMA = (
    "prismaquant.rtx4090_two_host_campaign.attempt_telemetry_record.v1"
)
_ATTEMPT_START_DISPOSITIONS = frozenset({
    "started",
    "adopted_after_start",
    "existing",
    "existing_after_conflict",
    "recovered_ambiguous_start",
})
_MISSING_PREFIX_START_DISPOSITIONS = frozenset({
    "existing",
    "existing_after_conflict",
    "recovered_ambiguous_start",
})
WORKER_SOURCE_CACHE_RECEIPT_SCHEMA = (
    "prismaquant.sample_parallel_probe.worker_source_cache_receipt.v1"
)

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
    "export_validation_artifact",
})
_RESUMABLE_FLAG_STAGES = frozenset({
    "measure_burn", "allocate", "export_validation_artifact",
})
_STAGE_TIMEOUT_SECONDS = {
    "host_preflight": 600.0,
    "prepare_calibration": 7200.0,
    "coordinator_source_identity": 21600.0,
    "prepare_run_contract": 3600.0,
    "build_sample_cover": 3600.0,
    "worker_source_identity": 21600.0,
    "sample_ce": 86400.0,
    "merge_importance": 7200.0,
    "sample_fisher": 86400.0,
    "merge_sample_probe": 21600.0,
    "derive_col_weights": 21600.0,
    "attest_execution": 3600.0,
    "prepare_burn": 21600.0,
    "measure_burn": 604800.0,
    "merge_burn": 21600.0,
    "allocate": 172800.0,
    "export_validation_artifact": 604800.0,
    "qualify_validation_artifact": 86400.0,
}
_CONTAINER_CLEANUP_GRACE_SECONDS = 600.0
_STAGE_DEPENDENCIES = {
    spec.stage: frozenset(spec.dependencies) for spec in STAGE_DAG
}
_LEGACY_EXECUTION_RECEIPT_KEYS = frozenset({
    "schema",
    "campaign_identity_sha256",
    "host_identity_sha256",
    "producer_identity_sha256",
    "work_id",
    "stage",
    "host_id",
    "assignment_index",
    "attempt_index",
    "command_identity_sha256",
    "request_sha256",
    "dependency_receipt_sha256s",
    "precondition_receipt_sha256s",
    "job",
    "gpu_bearing",
    "gpu_activity_requirement",
    "telemetry",
    "telemetry_sha256",
    "telemetry_summary",
    "output_artifacts",
    "identity_sha256",
})
_EXECUTION_RECEIPT_KEYS = _LEGACY_EXECUTION_RECEIPT_KEYS | {
    "attempt_telemetry_identity_sha256",
}
_OUTPUT_ARTIFACT_KEYS = frozenset({
    "container_path",
    "host_path",
    "expected_kind",
    "manifest",
    "manifest_sha256",
})
_LEGACY_ATTEMPT_TELEMETRY_KEYS = frozenset({
    "schema",
    "campaign_identity_sha256",
    "work_id",
    "attempt_index",
    "request_sha256",
    "samples",
    "identity_sha256",
})
_LEGACY_ATTEMPT_TELEMETRY_V2_KEYS = _LEGACY_ATTEMPT_TELEMETRY_KEYS | {
    "sampling_failure_count",
    "sampling_integrity_failure_count",
    "consecutive_sampling_failure_count",
    "maximum_consecutive_sampling_failure_count",
    "missing_prior_journal",
}
_ATTEMPT_TELEMETRY_HEAD_KEYS = frozenset({
    "schema",
    "campaign_identity_sha256",
    "work_id",
    "attempt_index",
    "request_sha256",
    "start_phase",
    "start_disposition",
    "next_ordinal",
    "record_count",
    "head_record_sha256",
    "pending_sample_ordinal",
    "last_sample_captured_ns",
    "sampling_failure_count",
    "sampling_integrity_failure_count",
    "consecutive_sampling_failure_count",
    "maximum_consecutive_sampling_failure_count",
    "missing_prior_journal",
    "identity_sha256",
})
_ATTEMPT_TELEMETRY_RECORD_KEYS = frozenset({
    "schema",
    "campaign_identity_sha256",
    "work_id",
    "attempt_index",
    "request_sha256",
    "ordinal",
    "previous_record_sha256",
    "outcome",
    "sample",
    "identity_sha256",
})
_ATTEMPT_TELEMETRY_STATE_KEYS = frozenset({
    "schema",
    "samples",
    "sampling_failure_count",
    "sampling_integrity_failure_count",
    "consecutive_sampling_failure_count",
    "maximum_consecutive_sampling_failure_count",
    "missing_prior_journal",
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


class _AttemptRetryRequired(RTX4090TwoHostCampaignError):
    """One exact attempt is unusable, but proven absence permits the next."""


class CampaignTransport(Protocol):
    """Durable execution seam used for launch, adoption, and monitoring."""

    def start(self, request: RunRequest) -> object:
        ...

    def status(self, job_id: str) -> object:
        ...

    def sample_telemetry(self) -> object:
        ...


class ArtifactInspector(Protocol):
    """Host-specific authority for one absolute output file or tree."""

    def inspect_artifact(self, absolute_path: str) -> object:
        ...


class StagePreconditioner(Protocol):
    """Persist or verify non-process receipts required before one stage."""

    def __call__(
        self, stage: str, *, verify_only: bool,
    ) -> Sequence[str]:
        ...


class LocalArtifactInspector:
    """No-follow local implementation of :class:`ArtifactInspector`."""

    def inspect_artifact(self, absolute_path: str) -> TreeManifest:
        return build_tree_manifest(absolute_path)


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


def _host_output_path(
    host: Mapping[str, object], container_path: str,
) -> tuple[str, str]:
    """Map one fixed mutable container output to its host-local pathname."""

    candidate = PurePosixPath(container_path)
    if not candidate.is_absolute() or str(candidate) != container_path:
        raise RTX4090TwoHostCampaignError(
            f"declared output is not a normalized absolute path: {container_path!r}"
        )
    roots = host["roots"]
    assert isinstance(roots, Mapping)
    for root_field in ("run_root", "worker_state_root"):
        container_root = PurePosixPath(CANONICAL_CONTAINER_PATHS[root_field])
        prefix = container_root.parts
        if candidate.parts[: len(prefix)] != prefix:
            continue
        relative = candidate.parts[len(prefix):]
        if not relative or any(part in {"", ".", ".."} for part in relative):
            raise RTX4090TwoHostCampaignError(
                f"declared output must be below {container_root}: {container_path}"
            )
        host_root = PurePosixPath(str(roots[root_field]))
        resolved = host_root.joinpath(*relative)
        if resolved.parts[: len(host_root.parts)] != host_root.parts:
            raise RTX4090TwoHostCampaignError(
                f"declared output escapes host root: {container_path}"
            )
        return resolved.as_posix(), root_field
    raise RTX4090TwoHostCampaignError(
        "declared outputs may exist only below /run or /worker-state: "
        f"{container_path}"
    )


def _expected_output_kind(container_path: str) -> str:
    name = PurePosixPath(container_path).name
    return (
        "directory"
        if name in {"act", "activation_cache", "validation-artifact"}
        else "file"
    )


def _normalize_tree_manifest(value: object, *, where: str) -> TreeManifest:
    try:
        manifest = (
            value
            if isinstance(value, TreeManifest)
            else TreeManifest.from_payload(value)
        )
    except (TypeError, ValueError) as exc:
        raise RTX4090TwoHostCampaignError(
            f"{where} returned an invalid file/tree manifest"
        ) from exc
    return manifest


def _inspect_command_outputs(
    command: CampaignCommand,
    host: Mapping[str, object],
    inspector: ArtifactInspector | None,
) -> list[dict[str, object]]:
    if not command.outputs:
        return []
    if inspector is None:
        raise RTX4090TwoHostCampaignError(
            f"no artifact inspector for output-bearing host {command.host_id!r}"
        )
    rows: list[dict[str, object]] = []
    seen_host_paths: set[str] = set()
    for container_path in command.outputs:
        host_path, _ = _host_output_path(host, container_path)
        if host_path in seen_host_paths:
            raise RTX4090TwoHostCampaignError(
                f"command {command.work_id} declares a duplicate output"
            )
        seen_host_paths.add(host_path)
        try:
            raw_manifest = inspector.inspect_artifact(host_path)
        except Exception as exc:
            raise RTX4090TwoHostCampaignError(
                f"declared output is absent or unverifiable: {host_path}"
            ) from exc
        manifest = _normalize_tree_manifest(
            raw_manifest, where=f"output {container_path}"
        )
        expected_kind = _expected_output_kind(container_path)
        if manifest.root_kind != expected_kind:
            raise RTX4090TwoHostCampaignError(
                f"declared output {container_path} must be a {expected_kind}"
            )
        rows.append({
            "container_path": container_path,
            "host_path": host_path,
            "expected_kind": expected_kind,
            "manifest": manifest.to_payload(),
            "manifest_sha256": manifest.identity_sha256,
        })
    return rows


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
    *, work_id: str, module: str, module_argv: Sequence[str], enable_gpu: bool,
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
        "docker", "run", "--rm",
        "--label", f"io.prismaquant.campaign={manifest['identity_sha256']}",
        "--label", f"io.prismaquant.host={host['id']}",
        "--label", f"io.prismaquant.work={work_id}",
    ]
    if enable_gpu:
        argv.extend(("--gpus", "all"))
    else:
        # The pinned image declares NVIDIA_VISIBLE_DEVICES=all.  Override it
        # explicitly so CPU-only stages stay GPU-blind even when the host's
        # Docker default runtime is NVIDIA rather than runc.
        argv.extend(("--env", "NVIDIA_VISIBLE_DEVICES=void"))
    argv.extend((
        "--ipc=host", "--uts=host",
        "--user", f"{expected['uid']}:{expected['gid']}",
        "--workdir", "/worker-state/tmp",
    ))
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


def _guarded_launch_argv(
    host: Mapping[str, object], docker_argv: Sequence[str],
    *, work_id: str, timeout_seconds: float,
) -> tuple[str, ...]:
    roots = host["roots"]
    assert isinstance(roots, Mapping)
    snapshot = _safe_mount_source(roots["snapshot_root"], field="snapshot_root")
    return (
        "python3", "-P", "-B", "-s",
        f"{snapshot}/tools/prismaquant_source_bootstrap.py",
        "run-module", "--source-root", snapshot,
        "prismaquant.cluster_host_admission", "guarded-launch",
        "--host-id", str(host["id"]),
        "--work-id", work_id,
        "--timeout-seconds", str(float(timeout_seconds)),
        "--", *map(str, docker_argv),
    )


def _stage_module_argv(
    manifest: Mapping[str, object], stage: str, index: int,
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
        ), (
            "/model", "/dataset", "/pq",
            "/run/coordinator/source-identity-cache.json",
        ), ("/run/run-contract.json",)
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
        ), (
            "/model", "/dataset", "/pq", "/run/calibration.pt",
            "/run/run-contract.json", "/run/cover.json",
        ), (
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
        ), (
            "/model", "/dataset", "/pq", "/run/calibration.pt",
            "/run/run-contract.json", "/run/cover.json",
            "/run/global-ce.json",
        ), (f"{worker}/probe.pkl", f"{worker}/act")
    if stage == "merge_sample_probe":
        return "prismaquant.sample_parallel_probe", (
            "merge", "--cover", "/run/cover.json", "--probe-shards",
            "/run/worker-0/probe.pkl", "/run/worker-1/probe.pkl",
            "--activation-cache", "0=/run/worker-0/act", "1=/run/worker-1/act",
            "--output-bundle", "/run/merged", "--max-rows", "1024",
        ), (
            "/run/cover.json",
            "/run/worker-0/probe.pkl", "/run/worker-1/probe.pkl",
            "/run/worker-0/act", "/run/worker-1/act",
        ), (
            "/run/merged/probe.pkl", "/run/merged/activation_cache", "/run/merged/commit.json",
        )
    if stage == "derive_col_weights":
        return "prismaquant.rtx4090_fp8_burn", (
            "derive-col-weights", "--sample-merge-bundle", "/run/merged",
            "--output", "/run/cb_col_weights.pkl",
        ), (
            "/run/merged/probe.pkl",
            "/run/merged/activation_cache",
            "/run/merged/commit.json",
        ), ("/run/cb_col_weights.pkl",)
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
        ), (
            "/model", "/dataset", "/pq",
            "/run/coordinator/source-identity-cache.json",
            "/run/merged/probe.pkl", "/run/merged/activation_cache",
            "/run/merged/commit.json", "/run/cb_col_weights.pkl",
            "/run/burn/execution-attestation.json",
        ), (
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
        ), (
            "/model", "/dataset", "/pq",
            "/worker-state/source-identity-cache.json",
            "/run/merged/probe.pkl", "/run/merged/activation_cache",
            "/run/merged/commit.json", "/run/cb_col_weights.pkl",
            "/run/burn/execution-attestation.json",
            "/run/burn/plan/campaign-plan.json",
        ), (
            f"/run/burn/stripe-{index}.pkl",
        )
    if stage == "merge_burn":
        return "prismaquant.rtx4090_fp8_burn", (
            "merge", "--plan", "/run/burn/plan/campaign-plan.json",
            "--producer-snapshot", SNAPSHOT_MANIFEST, "--col-weights",
            "/run/cb_col_weights.pkl", "--shards", "/run/burn/stripe-0.pkl",
            "/run/burn/stripe-1.pkl", "--output", "/run/burn/aura-merged.pkl",
        ), (
            "/pq", "/run/burn/plan/campaign-plan.json",
            "/run/cb_col_weights.pkl",
            "/run/burn/stripe-0.pkl", "/run/burn/stripe-1.pkl",
        ), (
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
        ), (
            "/model", "/pq", "/run/burn/plan/campaign-plan.json",
            "/run/merged/probe.pkl", "/run/merged/activation_cache",
            "/run/merged/commit.json", "/run/cb_col_weights.pkl",
            "/run/burn/aura-merged.pkl",
        ), (
            "/run/burn/allocator-cost.pkl", "/run/burn/allocation/layer_config.json",
        )
    if stage == "export_validation_artifact":
        inputs = manifest["inputs"]
        assert isinstance(inputs, Mapping)
        contract_input = inputs["gridbook_runtime_contract"]
        assert isinstance(contract_input, Mapping)
        contract_payload = json.dumps(
            contract_input["payload"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        contract_base64 = base64.b64encode(contract_payload).decode("ascii")
        return "prismaquant.rtx4090_validation_export", (
            "--model-dir", "/model",
            "--layer-config", "/run/burn/allocation/layer_config.json",
            "--col-weights", "/run/cb_col_weights.pkl",
            "--runtime-contract-payload-base64", contract_base64,
            "--runtime-contract-sha256",
            str(contract_input["runtime_contract_sha256"]),
            "--runtime-contract-output",
            "/run/burn/gridbook-runtime-contract.json",
            "--out", "/run/validation-artifact",
            "--shard-bytes", "1073741824",
            "--export-only",
        ), (
            "/model", "/pq", "/run/burn/allocation/layer_config.json",
            "/run/cb_col_weights.pkl",
        ), (
            "/run/burn/gridbook-runtime-contract.json",
            "/run/validation-artifact",
        )
    if stage == "qualify_validation_artifact":
        return "prismaquant.validate_rtx4090_fp8_cb_validation_only", (
            "/run/validation-artifact",
            "--runtime-contract", "/run/burn/gridbook-runtime-contract.json",
            "--receipt", "/run/burn/validation-package-receipt.json",
        ), (
            "/run/validation-artifact",
            "/run/burn/gridbook-runtime-contract.json",
        ), (
            "/run/burn/validation-package-receipt.json",
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
        normalized, assignment.stage, worker_index,
    )
    module_argv = tuple(
        str(producer["image_digest"]) if value == "__IMAGE_DIGEST__" else value
        for value in raw_module_argv
    )
    gpu_bearing = assignment.stage in _GPU_STAGES
    docker_argv = _docker_argv(
        normalized, host, work_id=assignment.work_id,
        module=module, module_argv=module_argv,
        enable_gpu=gpu_bearing,
    )
    stage_timeout = _STAGE_TIMEOUT_SECONDS[assignment.stage]
    argv = _guarded_launch_argv(
        host,
        docker_argv,
        work_id=assignment.work_id,
        timeout_seconds=stage_timeout,
    )
    resume_argv = None
    if assignment.stage in _RESUMABLE_FLAG_STAGES:
        resume_module_argv = (*module_argv, "--resume")
        resume_argv = _guarded_launch_argv(
            host,
            _docker_argv(
                normalized, host, work_id=assignment.work_id,
                module=module, module_argv=resume_module_argv,
                enable_gpu=gpu_bearing,
            ),
            work_id=assignment.work_id,
            timeout_seconds=stage_timeout,
        )
    return CampaignCommand(
        work_id=assignment.work_id, stage=assignment.stage,
        host_id=assignment.host_id, assignment_index=assignment.assignment_index,
        worker_index=worker_index,
        module=module, module_argv=module_argv, argv=argv,
        resume_argv=resume_argv, env=_HOST_ENV, inputs=inputs, outputs=outputs,
        gpu_bearing=gpu_bearing,
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


def _strict_json_mapping(path: Path, *, where: str) -> dict[str, object]:
    def unique_object(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RTX4090TwoHostCampaignError(
                    f"{where} contains duplicate JSON member {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise RTX4090TwoHostCampaignError(
            f"{where} contains non-JSON constant {value}"
        )

    try:
        payload = read_regular_file_nofollow(path, where=where)
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except RTX4090TwoHostCampaignError:
        raise
    except (
        ClusterTransportError, UnicodeDecodeError, json.JSONDecodeError,
    ) as exc:
        raise RTX4090TwoHostCampaignError(f"{where} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise RTX4090TwoHostCampaignError(f"{where} is not an object")
    normalized = dict(value)
    if payload != _json_bytes(normalized):
        raise RTX4090TwoHostCampaignError(
            f"{where} is not canonically encoded JSON"
        )
    return normalized


def _write_exact_no_clobber(path: Path, value: Mapping[str, object]) -> None:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise RTX4090TwoHostCampaignError(
            f"existing campaign artifact is unreadable: {path}"
        ) from exc
    if existing is not None:
        try:
            existing_payload = (
                read_regular_file_nofollow(
                    path, where="existing campaign artifact",
                )
                if stat.S_ISREG(existing.st_mode) else None
            )
        except ClusterTransportError as exc:
            raise RTX4090TwoHostCampaignError(
                f"existing campaign artifact is unsafe: {path}"
            ) from exc
        if existing_payload == payload:
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
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            try:
                raced = read_regular_file_nofollow(
                    path, where="racing campaign artifact",
                )
            except ClusterTransportError as read_error:
                raise RTX4090TwoHostCampaignError(
                    f"racing campaign artifact is unsafe: {path}"
                ) from read_error
            if raced != payload:
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


def _normalize_job_receipt(
    value: object,
    request: RunRequest,
    *,
    require_succeeded: bool = True,
) -> dict[str, object]:
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
    if require_succeeded and (
        result.get("state") != "succeeded" or result.get("returncode") != 0
    ):
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


def _strict_stdout_json(payload: bytes) -> Mapping[str, object] | None:
    """Parse the one-line machine receipt used for a telemetry waiver."""

    if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        return None

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON member")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload[:-1].decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _gpu_activity_requirement(
    manifest: Mapping[str, object],
    command: CampaignCommand,
    normalized_job: Mapping[str, object],
    *,
    attempt_index: int,
    legacy_retry_waiver: bool = False,
) -> str:
    """Derive the utilization rule from fixed work and byte-exact output.

    Source-identity cache validation deliberately returns before CUDA when an
    already-published cache validates exactly.  That is useful resumability,
    not missing GPU work.  Only that sealed disposition may waive positive
    utilization; a fresh source build and every numerical GPU stage retain the
    campaign's positive-activity requirement.
    """

    if not command.gpu_bearing:
        return "not_applicable"
    policy = manifest["policy"]
    assert isinstance(policy, Mapping)
    telemetry_policy = policy["telemetry"]
    assert isinstance(telemetry_policy, Mapping)
    if not bool(telemetry_policy["require_positive_gpu_utilization"]):
        return "telemetry_policy_disabled"
    if command.stage not in {
        "coordinator_source_identity", "worker_source_identity",
    }:
        return "positive_utilization_required"
    if attempt_index != 0 and not legacy_retry_waiver:
        # A failed first attempt may have created the cache without producing
        # acceptable telemetry.  Its retry must not turn that unobserved work
        # into a zero-utilization validated-reuse waiver.
        return "positive_utilization_required"
    try:
        output = JobReceipt.from_payload(normalized_job).stdout
    except (TypeError, ValueError):
        return "positive_utilization_required"
    receipt = _strict_stdout_json(output)
    if receipt is None or set(receipt) != {
        "schema", "disposition", "model", "cache", "cache_sha256",
        "identity", "portable_identity",
    }:
        return "positive_utilization_required"
    portable = receipt.get("portable_identity")
    inputs = manifest["inputs"]
    assert isinstance(inputs, Mapping)
    if (
        receipt.get("schema") != WORKER_SOURCE_CACHE_RECEIPT_SCHEMA
        or receipt.get("disposition") != "validated_reuse"
        or receipt.get("model") != "/model"
        or not isinstance(receipt.get("cache"), str)
        or _HEX64.fullmatch(str(receipt.get("cache_sha256"))) is None
        or not isinstance(receipt.get("identity"), Mapping)
        or not isinstance(portable, Mapping)
        or portable.get("schema")
        != "prismaquant.streamed_model.portable_content.v1"
        or portable.get("portable_content_sha256")
        != inputs["model_content_sha256"]
    ):
        return "positive_utilization_required"
    expected_cache = (
        "/run/coordinator/source-identity-cache.json"
        if command.stage == "coordinator_source_identity"
        else "/worker-state/source-identity-cache.json"
    )
    if receipt["cache"] != expected_cache:
        return "positive_utilization_required"
    return "waived_validated_source_cache_reuse"


def _nonnegative_telemetry_counter(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RTX4090TwoHostCampaignError(f"{where} is invalid")
    return value


def _normalize_attempt_telemetry_evidence(
    manifest: Mapping[str, object],
    command: CampaignCommand,
    normalized_job: Mapping[str, object],
    attempt_telemetry: Mapping[str, object] | None,
    *,
    activity_requirement: str,
    allow_legacy_v2: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object], str | None]:
    """Bind one final job receipt to its exact durable telemetry journal.

    Transient management-query failures are compatible with acceptance only
    when the sealed success-ratio and maximum-blind-window policy still holds.
    Integrity failures and a missing journal for an adopted job always fail the
    attempt after the child reaches durable terminal state.
    """

    policy = manifest["policy"]
    assert isinstance(policy, Mapping)
    telemetry_policy = policy["telemetry"]
    assert isinstance(telemetry_policy, Mapping)
    maximum_gap_limit_ns = (
        int(telemetry_policy["maximum_observation_gap_milliseconds"])
        * 1_000_000
    )
    minimum_success_percent = int(
        telemetry_policy["minimum_successful_sample_percent"]
    )

    if not command.gpu_bearing:
        if attempt_telemetry is not None:
            raise RTX4090TwoHostCampaignError(
                "CPU assignment unexpectedly has an attempt telemetry journal"
            )
        host = _host_by_id(manifest, command.host_id)
        expected = host["expected"]
        assert isinstance(expected, Mapping)
        expected_gpu = expected["gpu"]
        assert isinstance(expected_gpu, Mapping)
        samples, summary = _normalize_telemetry(
            (), required=False, expected_gpu=expected_gpu,
        )
        return samples, {
            **summary,
            "sampling_attempt_count": 0,
            "successful_sample_count": 0,
            "sampling_failure_count": 0,
            "sampling_integrity_failure_count": 0,
            "outside_job_lifetime_sample_count": 0,
            "unobservable_gpu_utilization_sample_count": 0,
            "consecutive_sampling_failure_count": 0,
            "maximum_consecutive_sampling_failure_count": 0,
            "successful_sample_percent": None,
            "maximum_observation_gap_ns": None,
            "maximum_observation_gap_limit_ns": maximum_gap_limit_ns,
            "minimum_successful_sample_percent_required": (
                minimum_success_percent
            ),
            "missing_prior_journal": False,
        }, None

    permitted_schemas = {ATTEMPT_TELEMETRY_SCHEMA}
    if allow_legacy_v2:
        permitted_schemas.add(LEGACY_ATTEMPT_TELEMETRY_V2_SCHEMA)
    if (
        not isinstance(attempt_telemetry, Mapping)
        or set(attempt_telemetry) != _ATTEMPT_TELEMETRY_STATE_KEYS
        or attempt_telemetry.get("schema") not in permitted_schemas
    ):
        raise RTX4090TwoHostCampaignError(
            "GPU assignment requires a current attempt telemetry journal"
        )
    journal_identity = attempt_telemetry.get("identity_sha256")
    if _HEX64.fullmatch(str(journal_identity)) is None:
        raise RTX4090TwoHostCampaignError(
            "GPU assignment attempt telemetry identity is malformed"
        )
    raw_samples = attempt_telemetry.get("samples")
    if not isinstance(raw_samples, list):
        raise RTX4090TwoHostCampaignError(
            "GPU assignment attempt telemetry samples are malformed"
        )
    transient_failures = _nonnegative_telemetry_counter(
        attempt_telemetry.get("sampling_failure_count"),
        where="telemetry sampling failure count",
    )
    integrity_failures = _nonnegative_telemetry_counter(
        attempt_telemetry.get("sampling_integrity_failure_count"),
        where="telemetry sampling integrity failure count",
    )
    consecutive_failures = _nonnegative_telemetry_counter(
        attempt_telemetry.get("consecutive_sampling_failure_count"),
        where="telemetry consecutive sampling failure count",
    )
    maximum_consecutive_failures = _nonnegative_telemetry_counter(
        attempt_telemetry.get("maximum_consecutive_sampling_failure_count"),
        where="telemetry maximum consecutive sampling failure count",
    )
    missing_prior_journal = attempt_telemetry.get("missing_prior_journal")
    if not isinstance(missing_prior_journal, bool):
        raise RTX4090TwoHostCampaignError(
            "telemetry missing-prior-journal marker is invalid"
        )
    total_failures = transient_failures + integrity_failures
    if (
        consecutive_failures > maximum_consecutive_failures
        or maximum_consecutive_failures > total_failures
        or ((total_failures == 0) != (maximum_consecutive_failures == 0))
        or (missing_prior_journal and integrity_failures < 1)
    ):
        raise RTX4090TwoHostCampaignError(
            "telemetry sampling counter relationships are invalid"
        )

    host = _host_by_id(manifest, command.host_id)
    expected = host["expected"]
    assert isinstance(expected, Mapping)
    expected_gpu = expected["gpu"]
    assert isinstance(expected_gpu, Mapping)
    all_samples, _ = _normalize_telemetry(
        raw_samples,
        required=False,
        expected_gpu=expected_gpu,
    )
    started_ns = normalized_job.get("started_ns")
    finished_ns = normalized_job.get("finished_ns")
    if (
        isinstance(started_ns, bool)
        or not isinstance(started_ns, int)
        or isinstance(finished_ns, bool)
        or not isinstance(finished_ns, int)
        or finished_ns < started_ns
    ):
        raise RTX4090TwoHostCampaignError(
            "final job receipt has an invalid telemetry observation window"
        )
    all_captures = [int(sample["captured_ns"]) for sample in all_samples]
    if any(value < started_ns for value in all_captures):
        raise RTX4090TwoHostCampaignError(
            "telemetry sample predates the final job lifetime"
        )
    samples = [
        sample for sample in all_samples
        if int(sample["captured_ns"]) <= finished_ns
    ]
    outside_job_lifetime_sample_count = len(all_samples) - len(samples)
    samples, summary = _normalize_telemetry(
        samples,
        required=activity_requirement == "positive_utilization_required",
        expected_gpu=expected_gpu,
    )
    observable_samples: list[dict[str, object]] = []
    for sample in samples:
        gpus = sample["gpus"]
        assert isinstance(gpus, list)
        matching_gpu = next(
            gpu for gpu in gpus
            if isinstance(gpu, Mapping)
            and gpu.get("name") == expected_gpu["name"]
            and gpu.get("uuid") == expected_gpu["uuid"]
        )
        if matching_gpu.get("gpu_utilization_pct") is not None:
            observable_samples.append(sample)
    unobservable_gpu_utilization_sample_count = (
        len(samples) - len(observable_samples)
    )
    captures = [
        int(sample["captured_ns"]) for sample in observable_samples
    ]
    observation_points = [started_ns, *captures, finished_ns]
    maximum_gap_ns = max(
        (right - left for left, right in zip(
            observation_points, observation_points[1:],
        )),
        default=0,
    )
    sampling_attempt_count = len(all_samples) + total_failures
    successful_sample_count = len(observable_samples)
    successful_sample_percent = (
        100.0 * successful_sample_count / sampling_attempt_count
        if sampling_attempt_count else None
    )
    evidence_summary = {
        **summary,
        "sampling_attempt_count": sampling_attempt_count,
        "successful_sample_count": successful_sample_count,
        "sampling_failure_count": transient_failures,
        "sampling_integrity_failure_count": integrity_failures,
        "outside_job_lifetime_sample_count": (
            outside_job_lifetime_sample_count
        ),
        "unobservable_gpu_utilization_sample_count": (
            unobservable_gpu_utilization_sample_count
        ),
        "consecutive_sampling_failure_count": consecutive_failures,
        "maximum_consecutive_sampling_failure_count": (
            maximum_consecutive_failures
        ),
        "successful_sample_percent": successful_sample_percent,
        "maximum_observation_gap_ns": maximum_gap_ns,
        "maximum_observation_gap_limit_ns": maximum_gap_limit_ns,
        "minimum_successful_sample_percent_required": minimum_success_percent,
        "missing_prior_journal": missing_prior_journal,
    }
    if integrity_failures or missing_prior_journal:
        raise RTX4090TwoHostCampaignError(
            "telemetry integrity evidence invalidates the completed attempt"
        )
    if activity_requirement == "positive_utilization_required":
        assert successful_sample_percent is not None
        if successful_sample_percent < minimum_success_percent:
            raise RTX4090TwoHostCampaignError(
                "telemetry successful sample percentage is below policy"
            )
        if maximum_gap_ns > maximum_gap_limit_ns:
            raise RTX4090TwoHostCampaignError(
                "telemetry maximum observation gap exceeds policy"
            )
    return samples, evidence_summary, str(journal_identity)


def _stage_receipt(
    manifest: Mapping[str, object], command: CampaignCommand,
    request: RunRequest, job_receipt: object,
    attempt_telemetry: Mapping[str, object] | None,
    dependency_receipt_sha256s: Sequence[str],
    precondition_receipt_sha256s: Sequence[str], *, attempt_index: int,
    artifact_inspector: ArtifactInspector | None,
) -> dict[str, object]:
    if manifest.get("schema") != CAMPAIGN_MANIFEST_SCHEMA:
        raise RTX4090TwoHostCampaignError(
            "legacy campaign manifests are read-only"
        )
    job = _normalize_job_receipt(job_receipt, request)
    host = _host_by_id(manifest, command.host_id)
    activity_requirement = _gpu_activity_requirement(
        manifest, command, job, attempt_index=attempt_index,
    )
    samples, telemetry_summary, attempt_telemetry_identity = (
        _normalize_attempt_telemetry_evidence(
            manifest,
            command,
            job,
            attempt_telemetry,
            activity_requirement=activity_requirement,
        )
    )
    output_artifacts = _inspect_command_outputs(
        command, host, artifact_inspector,
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
        "attempt_index": attempt_index,
        "command_identity_sha256": command.identity_sha256,
        "request_sha256": request.request_sha256,
        "dependency_receipt_sha256s": list(dependency_receipt_sha256s),
        "precondition_receipt_sha256s": list(precondition_receipt_sha256s),
        "job": job,
        "gpu_bearing": command.gpu_bearing,
        "gpu_activity_requirement": activity_requirement,
        "telemetry": samples,
        "telemetry_sha256": canonical_sha256(samples),
        "telemetry_summary": telemetry_summary,
        "attempt_telemetry_identity_sha256": attempt_telemetry_identity,
        "output_artifacts": output_artifacts,
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


class CampaignCoordinator:
    def __init__(
        self,
        manifest: Mapping[str, object],
        state_dir: str | Path,
        *,
        transports: Mapping[str, CampaignTransport] | None = None,
        artifact_inspectors: Mapping[str, ArtifactInspector] | None = None,
        stage_preconditioner: StagePreconditioner | None = None,
        sleep: Callable[[float], None] = time.sleep,
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
        self.artifact_inspectors = dict(artifact_inspectors or {})
        self.stage_preconditioner = stage_preconditioner
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self._sleep = sleep
        policy = self.manifest["policy"]
        assert isinstance(policy, Mapping)
        retry_policy = policy["retry"]
        telemetry_policy = policy["telemetry"]
        assert isinstance(retry_policy, Mapping)
        assert isinstance(telemetry_policy, Mapping)
        self._max_attempts = int(retry_policy["max_attempts"])
        self._telemetry_interval_seconds = (
            int(telemetry_policy["interval_milliseconds"]) / 1000.0
        )
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

    @property
    def max_attempts(self) -> int:
        """Return the sealed, bounded attempt count used by every command."""

        return self._max_attempts

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
        self._require_current_manifest("initialize")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        _write_exact_no_clobber(
            self.state_dir / MANIFEST_FILE, self.manifest,
        )
        _write_exact_no_clobber(
            self.state_dir / PLAN_FILE, build_command_plan(self.manifest),
        )
        try:
            self.state_path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise RTX4090TwoHostCampaignError(
                "campaign state already exists; use resume"
            )
        state = initial_campaign_state(self.manifest)
        _replace_state(self.state_path, state)
        return state

    def load_state(self) -> dict[str, object]:
        try:
            state_metadata = self.state_path.lstat()
        except FileNotFoundError:
            raise RTX4090TwoHostCampaignError("campaign state is absent; use run")
        except OSError as exc:
            raise RTX4090TwoHostCampaignError(
                "campaign state is unreadable"
            ) from exc
        if not stat.S_ISREG(state_metadata.st_mode):
            raise RTX4090TwoHostCampaignError(
                "campaign state is not a real regular file"
            )
        raw = _strict_json_mapping(
            self.state_path, where="campaign state",
        )
        return validate_campaign_state(raw, self.manifest)

    def _require_current_manifest(self, operation: str) -> None:
        if self.manifest["schema"] == LEGACY_CAMPAIGN_MANIFEST_SCHEMA:
            raise RTX4090TwoHostCampaignError(
                f"legacy campaign manifests are read-only; cannot {operation}"
            )

    def _request(
        self, command: CampaignCommand, *, attempt_index: int,
        precondition_receipt_sha256s: Sequence[str] = (),
    ) -> RunRequest:
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or not 0 <= attempt_index < self._max_attempts
        ):
            raise RTX4090TwoHostCampaignError(
                "attempt index is outside the fixed bounded retry policy"
            )
        argv = (
            command.resume_argv
            if attempt_index > 0 and command.resume_argv is not None
            else command.argv
        )
        campaign_prefix = str(self.manifest["identity_sha256"])[:16]
        host_token = command.host_id
        job_id = (
            f"pq-{campaign_prefix}-{command.assignment_index:02d}-"
            f"{command.stage}-{host_token}-attempt-{attempt_index:02d}"
        )
        if len(job_id) > 128:
            host_token = "host-" + hashlib.sha256(
                command.host_id.encode("utf-8")
            ).hexdigest()[:16]
            job_id = (
                f"pq-{campaign_prefix}-{command.assignment_index:02d}-"
                f"{command.stage}-{host_token}-attempt-{attempt_index:02d}"
            )
        preconditions = self._normalize_precondition_sha256s(
            precondition_receipt_sha256s,
        )
        request_env = dict(command.env)
        request_env["PQ_CAMPAIGN_PRECONDITIONS_SHA256"] = canonical_sha256(
            list(preconditions)
        )
        request_env["PQ_CAMPAIGN_GPU_BEARING"] = (
            "1" if command.gpu_bearing else "0"
        )
        request_env["PQ_CAMPAIGN_START_GUARD"] = (
            "1" if command.module is not None else "0"
        )
        return RunRequest(
            job_id=job_id,
            argv=argv,
            cwd="/",
            env=tuple(sorted(request_env.items())),
            timeout_seconds=(
                _STAGE_TIMEOUT_SECONDS[command.stage]
                + (
                    _CONTAINER_CLEANUP_GRACE_SECONDS
                    if command.module is not None else 0.0
                )
            ),
            stdin=(
                canonical_json_bytes(self.manifest)
                if command.module is not None else b""
            ),
            inherit_env=False,
        )

    @staticmethod
    def _normalize_precondition_sha256s(
        values: Sequence[str],
    ) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise RTX4090TwoHostCampaignError(
                "stage preconditions must be a digest sequence"
            )
        normalized = tuple(str(value) for value in values)
        if (
            normalized != tuple(sorted(set(normalized)))
            or any(_HEX64.fullmatch(value) is None for value in normalized)
        ):
            raise RTX4090TwoHostCampaignError(
                "stage precondition digests must be sorted unique SHA-256 values"
            )
        return normalized

    def _stage_preconditions(
        self, stage: str, *, verify_only: bool,
    ) -> tuple[str, ...]:
        if not verify_only:
            self._require_current_manifest("evaluate mutating stage preconditions")
        if self.stage_preconditioner is None:
            return ()
        try:
            values = self.stage_preconditioner(
                stage, verify_only=verify_only,
            )
        except Exception as exc:
            raise RTX4090TwoHostCampaignError(
                f"stage preconditions failed for {stage}: {exc}"
            ) from exc
        return self._normalize_precondition_sha256s(values)

    def _attempt_telemetry_path(self, request: RunRequest) -> Path:
        return (
            self.state_dir / ATTEMPT_TELEMETRY_DIR /
            f"{request.job_id}.json"
        )

    def _ensure_attempt_telemetry_root(self) -> Path:
        root = self.state_dir / ATTEMPT_TELEMETRY_DIR
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if root.is_symlink() or not root.is_dir():
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry root is unsafe"
            )
        parent_descriptor = os.open(root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return root

    def _attempt_telemetry_records_dir(self, request: RunRequest) -> Path:
        return (
            self.state_dir / ATTEMPT_TELEMETRY_DIR /
            f"{request.job_id}.records"
        )

    def _attempt_telemetry_record_path(
        self, request: RunRequest, ordinal: int,
    ) -> Path:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry record ordinal is invalid"
            )
        return self._attempt_telemetry_records_dir(request) / f"{ordinal:020d}.json"

    def _ensure_attempt_telemetry_records_dir(self, request: RunRequest) -> Path:
        self._ensure_attempt_telemetry_root()
        root = self._attempt_telemetry_records_dir(request)
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if root.is_symlink() or not root.is_dir():
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry record root is unsafe"
            )
        parent_descriptor = os.open(root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return root

    @staticmethod
    def _validate_attempt_telemetry_counters(
        sampling_failure_count: int,
        sampling_integrity_failure_count: int,
        consecutive_sampling_failure_count: int,
        maximum_consecutive_sampling_failure_count: int,
        missing_prior_journal: bool,
    ) -> None:
        counts = (
            sampling_failure_count,
            sampling_integrity_failure_count,
            consecutive_sampling_failure_count,
            maximum_consecutive_sampling_failure_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise RTX4090TwoHostCampaignError(
                "telemetry sampling counters are invalid"
            )
        failed_attempts = sampling_failure_count + sampling_integrity_failure_count
        if (
            consecutive_sampling_failure_count
            > maximum_consecutive_sampling_failure_count
            or maximum_consecutive_sampling_failure_count > failed_attempts
            or ((failed_attempts == 0) != (maximum_consecutive_sampling_failure_count == 0))
            or not isinstance(missing_prior_journal, bool)
            or (missing_prior_journal and sampling_integrity_failure_count < 1)
        ):
            raise RTX4090TwoHostCampaignError(
                "telemetry sampling counter relationships are invalid"
            )

    def _attempt_telemetry_head_payload(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        *,
        start_phase: str,
        start_disposition: str | None,
        next_ordinal: int,
        record_count: int,
        head_record_sha256: str | None,
        pending_sample_ordinal: int | None,
        last_sample_captured_ns: int | None,
        sampling_failure_count: int,
        sampling_integrity_failure_count: int,
        consecutive_sampling_failure_count: int,
        maximum_consecutive_sampling_failure_count: int,
        missing_prior_journal: bool,
    ) -> dict[str, object]:
        self._validate_attempt_telemetry_counters(
            sampling_failure_count,
            sampling_integrity_failure_count,
            consecutive_sampling_failure_count,
            maximum_consecutive_sampling_failure_count,
            missing_prior_journal,
        )
        disposition_phase = start_phase in {"claim_pending", "claimed"}
        missing_prefix_disposition = (
            start_disposition in _MISSING_PREFIX_START_DISPOSITIONS
        )
        if (
            start_phase not in {
                "prepared", "start_invoked", "claim_pending", "claimed",
            }
            or (
                disposition_phase
                != isinstance(start_disposition, str)
            )
            or (
                isinstance(start_disposition, str)
                and start_disposition not in _ATTEMPT_START_DISPOSITIONS
            )
            or (
                start_phase == "claim_pending"
                and not missing_prefix_disposition
            )
            or isinstance(next_ordinal, bool)
            or not isinstance(next_ordinal, int)
            or next_ordinal < 0
            or isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count != next_ordinal
            or (
                (record_count == 0)
                != (head_record_sha256 is None)
            )
            or (
                head_record_sha256 is not None
                and _HEX64.fullmatch(head_record_sha256) is None
            )
            or (
                pending_sample_ordinal is not None
                and (
                    isinstance(pending_sample_ordinal, bool)
                    or not isinstance(pending_sample_ordinal, int)
                    or pending_sample_ordinal != next_ordinal
                )
            )
            or (
                last_sample_captured_ns is not None
                and (
                    isinstance(last_sample_captured_ns, bool)
                    or not isinstance(last_sample_captured_ns, int)
                    or last_sample_captured_ns <= 0
                )
            )
            or (
                start_phase == "claim_pending"
                and pending_sample_ordinal is None
                and (
                    not missing_prior_journal
                    or sampling_integrity_failure_count < 1
                    or record_count < 1
                )
            )
            or (
                start_phase == "claimed"
                and missing_prefix_disposition
                and (
                    not missing_prior_journal
                    or sampling_integrity_failure_count < 1
                    or record_count < 1
                )
            )
        ):
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry head fields are invalid"
            )
        body: dict[str, object] = {
            "schema": ATTEMPT_TELEMETRY_SCHEMA,
            "campaign_identity_sha256": self.manifest["identity_sha256"],
            "work_id": command.work_id,
            "attempt_index": attempt_index,
            "request_sha256": request.request_sha256,
            "start_phase": start_phase,
            "start_disposition": start_disposition,
            "next_ordinal": next_ordinal,
            "record_count": record_count,
            "head_record_sha256": head_record_sha256,
            "pending_sample_ordinal": pending_sample_ordinal,
            "last_sample_captured_ns": last_sample_captured_ns,
            "sampling_failure_count": sampling_failure_count,
            "sampling_integrity_failure_count": sampling_integrity_failure_count,
            "consecutive_sampling_failure_count": consecutive_sampling_failure_count,
            "maximum_consecutive_sampling_failure_count": (
                maximum_consecutive_sampling_failure_count
            ),
            "missing_prior_journal": missing_prior_journal,
        }
        return {**body, "identity_sha256": canonical_sha256(body)}

    def _load_attempt_telemetry_head(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
    ) -> dict[str, object]:
        path = self._attempt_telemetry_path(request)
        if path.is_symlink():
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry head is a symlink"
            )
        raw = _strict_json_mapping(
            path, where=f"attempt telemetry for {command.work_id}",
        )
        if raw.get("schema") != ATTEMPT_TELEMETRY_SCHEMA:
            raise RTX4090TwoHostCampaignError(
                "legacy attempt telemetry has no mutable head"
            )
        if set(raw) != _ATTEMPT_TELEMETRY_HEAD_KEYS:
            raise RTX4090TwoHostCampaignError(
                f"attempt telemetry fields differ for {command.work_id}"
            )
        body = dict(raw)
        identity = body.pop("identity_sha256", None)
        if identity != canonical_sha256(body):
            raise RTX4090TwoHostCampaignError(
                f"attempt telemetry digest differs for {command.work_id}"
            )
        exact = {
            "campaign_identity_sha256": self.manifest["identity_sha256"],
            "work_id": command.work_id,
            "attempt_index": attempt_index,
            "request_sha256": request.request_sha256,
        }
        for key, expected_value in exact.items():
            if raw.get(key) != expected_value:
                raise RTX4090TwoHostCampaignError(
                    f"attempt telemetry {key} differs for {command.work_id}"
                )
        expected = self._attempt_telemetry_head_payload(
            command,
            request,
            attempt_index,
            start_phase=raw["start_phase"],  # type: ignore[arg-type]
            start_disposition=raw["start_disposition"],  # type: ignore[arg-type]
            next_ordinal=raw["next_ordinal"],  # type: ignore[arg-type]
            record_count=raw["record_count"],  # type: ignore[arg-type]
            head_record_sha256=raw["head_record_sha256"],  # type: ignore[arg-type]
            pending_sample_ordinal=raw["pending_sample_ordinal"],  # type: ignore[arg-type]
            last_sample_captured_ns=raw["last_sample_captured_ns"],  # type: ignore[arg-type]
            sampling_failure_count=raw["sampling_failure_count"],  # type: ignore[arg-type]
            sampling_integrity_failure_count=raw[
                "sampling_integrity_failure_count"
            ],  # type: ignore[arg-type]
            consecutive_sampling_failure_count=raw[
                "consecutive_sampling_failure_count"
            ],  # type: ignore[arg-type]
            maximum_consecutive_sampling_failure_count=raw[
                "maximum_consecutive_sampling_failure_count"
            ],  # type: ignore[arg-type]
            missing_prior_journal=raw["missing_prior_journal"],  # type: ignore[arg-type]
        )
        if expected != raw:
            raise RTX4090TwoHostCampaignError(
                f"attempt telemetry head is noncanonical for {command.work_id}"
            )
        return raw

    def _replace_attempt_telemetry_head(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        head: Mapping[str, object],
        **changes: object,
    ) -> dict[str, object]:
        values = {key: head[key] for key in _ATTEMPT_TELEMETRY_HEAD_KEYS if key not in {
            "schema", "campaign_identity_sha256", "work_id", "attempt_index",
            "request_sha256", "identity_sha256",
        }}
        values.update(changes)
        payload = self._attempt_telemetry_head_payload(
            command, request, attempt_index, **values,  # type: ignore[arg-type]
        )
        self._ensure_attempt_telemetry_root()
        _replace_state(self._attempt_telemetry_path(request), payload)
        return payload

    def _initialize_attempt_telemetry(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
    ) -> None:
        payload = self._attempt_telemetry_head_payload(
            command,
            request,
            attempt_index,
            start_phase="prepared",
            start_disposition=None,
            next_ordinal=0,
            record_count=0,
            head_record_sha256=None,
            pending_sample_ordinal=None,
            last_sample_captured_ns=None,
            sampling_failure_count=0,
            sampling_integrity_failure_count=0,
            consecutive_sampling_failure_count=0,
            maximum_consecutive_sampling_failure_count=0,
            missing_prior_journal=False,
        )
        self._ensure_attempt_telemetry_root()
        _write_exact_no_clobber(
            self._attempt_telemetry_path(request), payload,
        )
        self._ensure_attempt_telemetry_records_dir(request)

    def _begin_attempt_telemetry_sample(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
    ) -> dict[str, object]:
        head = self._load_attempt_telemetry_head(command, request, attempt_index)
        if head["pending_sample_ordinal"] is not None:
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry already has a pending sample"
            )
        return self._replace_attempt_telemetry_head(
            command,
            request,
            attempt_index,
            head,
            pending_sample_ordinal=head["next_ordinal"],
        )

    def _attempt_telemetry_record_payload(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        *,
        ordinal: int,
        previous_record_sha256: str | None,
        outcome: str,
        sample: Mapping[str, object] | None,
    ) -> dict[str, object]:
        if outcome not in {"sample", "unavailable", "integrity"}:
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry record outcome is invalid"
            )
        if (outcome == "sample") != isinstance(sample, Mapping):
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry record sample is invalid"
            )
        if (
            ordinal == 0 and previous_record_sha256 is not None
        ) or (
            ordinal > 0
            and (
                not isinstance(previous_record_sha256, str)
                or _HEX64.fullmatch(previous_record_sha256) is None
            )
        ):
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry record predecessor is invalid"
            )
        body: dict[str, object] = {
            "schema": ATTEMPT_TELEMETRY_RECORD_SCHEMA,
            "campaign_identity_sha256": self.manifest["identity_sha256"],
            "work_id": command.work_id,
            "attempt_index": attempt_index,
            "request_sha256": request.request_sha256,
            "ordinal": ordinal,
            "previous_record_sha256": previous_record_sha256,
            "outcome": outcome,
            "sample": None if sample is None else dict(sample),
        }
        return {**body, "identity_sha256": canonical_sha256(body)}

    def _read_attempt_telemetry_record(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        ordinal: int,
        previous_record_sha256: str | None,
    ) -> dict[str, object]:
        path = self._attempt_telemetry_record_path(request, ordinal)
        if path.is_symlink():
            raise RTX4090TwoHostCampaignError(
                f"attempt telemetry record {ordinal} is a symlink"
            )
        raw = _strict_json_mapping(
            path,
            where=f"attempt telemetry record {ordinal} for {command.work_id}",
        )
        if set(raw) != _ATTEMPT_TELEMETRY_RECORD_KEYS:
            raise RTX4090TwoHostCampaignError(
                f"attempt telemetry record {ordinal} fields differ"
            )
        expected = self._attempt_telemetry_record_payload(
            command,
            request,
            attempt_index,
            ordinal=ordinal,
            previous_record_sha256=previous_record_sha256,
            outcome=raw.get("outcome"),  # type: ignore[arg-type]
            sample=raw.get("sample"),  # type: ignore[arg-type]
        )
        if raw != expected:
            raise RTX4090TwoHostCampaignError(
                f"attempt telemetry record {ordinal} digest or binding differs"
            )
        return raw

    def _commit_attempt_telemetry_record(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        head: Mapping[str, object],
        record: Mapping[str, object],
        *,
        missing_prior_journal: bool,
    ) -> dict[str, object]:
        outcome = str(record["outcome"])
        failures = int(head["sampling_failure_count"])
        integrity = int(head["sampling_integrity_failure_count"])
        prior_consecutive = int(head["consecutive_sampling_failure_count"])
        maximum = int(head["maximum_consecutive_sampling_failure_count"])
        last_capture = head["last_sample_captured_ns"]
        if outcome == "sample":
            consecutive = 0
            sample = record["sample"]
            assert isinstance(sample, Mapping)
            last_capture = int(sample["captured_ns"])
        else:
            failures += int(outcome == "unavailable")
            integrity += int(outcome == "integrity")
            consecutive = prior_consecutive + 1
            maximum = max(maximum, consecutive)
        return self._replace_attempt_telemetry_head(
            command,
            request,
            attempt_index,
            head,
            next_ordinal=int(head["next_ordinal"]) + 1,
            record_count=int(head["record_count"]) + 1,
            head_record_sha256=record["identity_sha256"],
            pending_sample_ordinal=None,
            last_sample_captured_ns=last_capture,
            sampling_failure_count=failures,
            sampling_integrity_failure_count=integrity,
            consecutive_sampling_failure_count=consecutive,
            maximum_consecutive_sampling_failure_count=maximum,
            missing_prior_journal=(
                bool(head["missing_prior_journal"])
                or missing_prior_journal
            ),
        )

    def _append_attempt_telemetry_record(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        *,
        outcome: str,
        sample: object,
        require_pending: bool,
        missing_prior_journal: bool = False,
    ) -> dict[str, object]:
        head = self._load_attempt_telemetry_head(command, request, attempt_index)
        ordinal = int(head["next_ordinal"])
        pending = head["pending_sample_ordinal"]
        if require_pending and pending != ordinal:
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry sample was not durably pending"
            )
        if not require_pending and pending is not None:
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry has an unresolved pending sample"
            )
        if not require_pending:
            # Every record, including an integrity/unavailable row written
            # outside a sample operation, first reserves its ordinal in the
            # head.  A crash after record publication can therefore be
            # adopted rather than rejected as an orphan forever.
            head = self._replace_attempt_telemetry_head(
                command,
                request,
                attempt_index,
                head,
                pending_sample_ordinal=ordinal,
            )
        normalized_sample: Mapping[str, object] | None = None
        if outcome == "sample":
            host = _host_by_id(self.manifest, command.host_id)
            expected = host["expected"]
            assert isinstance(expected, Mapping)
            expected_gpu = expected["gpu"]
            assert isinstance(expected_gpu, Mapping)
            normalized, _ = _normalize_telemetry(
                [sample], required=False, expected_gpu=expected_gpu,
            )
            normalized_sample = normalized[0]
            prior_capture = head["last_sample_captured_ns"]
            if (
                prior_capture is not None
                and int(normalized_sample["captured_ns"]) <= int(prior_capture)
            ):
                raise RTX4090TwoHostCampaignError(
                    "attempt telemetry samples are not strictly increasing"
                )
        record = self._attempt_telemetry_record_payload(
            command,
            request,
            attempt_index,
            ordinal=ordinal,
            previous_record_sha256=head["head_record_sha256"],  # type: ignore[arg-type]
            outcome=outcome,
            sample=normalized_sample,
        )
        self._ensure_attempt_telemetry_records_dir(request)
        _write_exact_no_clobber(
            self._attempt_telemetry_record_path(request, ordinal), record,
        )
        return self._commit_attempt_telemetry_record(
            command,
            request,
            attempt_index,
            head,
            record,
            missing_prior_journal=missing_prior_journal,
        )

    def _recover_attempt_telemetry(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
    ) -> dict[str, object]:
        """Fail-latch interrupted sample/start write-ahead phases on resume."""

        head = self._load_attempt_telemetry_head(command, request, attempt_index)
        if head["start_phase"] == "start_invoked":
            # The process may have crossed start(2) before the coordinator
            # crashed.  Publish one durable intent that both reserves the
            # record ordinal and records why that integrity row must latch a
            # missing prefix.  No claimed state is legal before this row is
            # committed into the head.
            head = self._replace_attempt_telemetry_head(
                command,
                request,
                attempt_index,
                head,
                start_phase="claim_pending",
                start_disposition="recovered_ambiguous_start",
                pending_sample_ordinal=head["next_ordinal"],
            )
        pending = head["pending_sample_ordinal"]
        if pending is not None:
            ordinal = int(pending)
            path = self._attempt_telemetry_record_path(request, ordinal)
            if path.exists():
                record = self._read_attempt_telemetry_record(
                    command,
                    request,
                    attempt_index,
                    ordinal,
                    head["head_record_sha256"],  # type: ignore[arg-type]
                )
                head = self._commit_attempt_telemetry_record(
                    command,
                    request,
                    attempt_index,
                    head,
                    record,
                    missing_prior_journal=(
                        head["start_phase"] == "claim_pending"
                    ),
                )
            else:
                head = self._append_attempt_telemetry_record(
                    command,
                    request,
                    attempt_index,
                    outcome="integrity",
                    sample=None,
                    require_pending=True,
                    missing_prior_journal=(
                        head["start_phase"] == "claim_pending"
                    ),
                )
        elif self._attempt_telemetry_record_path(
            request, int(head["next_ordinal"]),
        ).exists():
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry has an uncommitted record without a "
                "pending write-ahead ordinal"
            )
        if head["start_phase"] == "claim_pending":
            if (
                not bool(head["missing_prior_journal"])
                or int(head["sampling_integrity_failure_count"]) < 1
            ):
                raise RTX4090TwoHostCampaignError(
                    "attempt start claim lacks its missing-prefix integrity row"
                )
            head = self._replace_attempt_telemetry_head(
                command,
                request,
                attempt_index,
                head,
                start_phase="claimed",
            )
        return head

    def _mark_attempt_start_invoked(
        self, command: CampaignCommand, request: RunRequest, attempt_index: int,
    ) -> None:
        head = self._load_attempt_telemetry_head(command, request, attempt_index)
        if head["start_phase"] != "prepared":
            raise RTX4090TwoHostCampaignError(
                "attempt start write-ahead phase is not pristine"
            )
        self._replace_attempt_telemetry_head(
            command, request, attempt_index, head,
            start_phase="start_invoked",
        )

    def _mark_attempt_start_claimed(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        disposition: str,
    ) -> None:
        head = self._load_attempt_telemetry_head(command, request, attempt_index)
        if head["start_phase"] not in {
            "prepared", "start_invoked", "claim_pending", "claimed",
        }:
            raise RTX4090TwoHostCampaignError(
                "attempt start write-ahead phase is invalid"
            )
        if head["start_phase"] == "claimed":
            if disposition == "existing":
                return
            if head["start_disposition"] != disposition:
                raise RTX4090TwoHostCampaignError(
                    "attempt start disposition differs on resume"
                )
            return
        if disposition not in _ATTEMPT_START_DISPOSITIONS:
            raise RTX4090TwoHostCampaignError(
                "attempt start disposition is invalid"
            )
        if head["start_phase"] == "claim_pending":
            if head["start_disposition"] != disposition:
                raise RTX4090TwoHostCampaignError(
                    "pending attempt start disposition differs on resume"
                )
            self._recover_attempt_telemetry(
                command, request, attempt_index,
            )
            return
        if disposition in _MISSING_PREFIX_START_DISPOSITIONS and not bool(
            head["missing_prior_journal"]
        ):
            head = self._replace_attempt_telemetry_head(
                command,
                request,
                attempt_index,
                head,
                start_phase="claim_pending",
                start_disposition=disposition,
                pending_sample_ordinal=head["next_ordinal"],
            )
            head = self._append_attempt_telemetry_record(
                command,
                request,
                attempt_index,
                outcome="integrity",
                sample=None,
                require_pending=True,
                missing_prior_journal=True,
            )
        self._replace_attempt_telemetry_head(
            command,
            request,
            attempt_index,
            head,
            start_phase="claimed",
            start_disposition=disposition,
        )

    def _load_attempt_telemetry(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        *,
        create_if_missing: bool = False,
    ) -> dict[str, object]:
        path = self._attempt_telemetry_path(request)
        if path.is_symlink():
            raise RTX4090TwoHostCampaignError(
                "attempt telemetry head is a symlink"
            )
        if not path.exists():
            if not create_if_missing:
                raise RTX4090TwoHostCampaignError(
                    f"attempt telemetry is absent for {command.work_id}"
                )
            self._initialize_attempt_telemetry(
                command, request, attempt_index,
            )
        raw = _strict_json_mapping(
            path, where=f"attempt telemetry for {command.work_id}",
        )
        schema = raw.get("schema")
        if schema == ATTEMPT_TELEMETRY_SCHEMA:
            head = self._load_attempt_telemetry_head(
                command, request, attempt_index,
            )
            samples: list[dict[str, object]] = []
            transient = 0
            integrity = 0
            consecutive = 0
            maximum = 0
            previous: str | None = None
            last_capture: int | None = None
            for ordinal in range(int(head["record_count"])):
                record = self._read_attempt_telemetry_record(
                    command, request, attempt_index, ordinal, previous,
                )
                previous = str(record["identity_sha256"])
                outcome = record["outcome"]
                if outcome == "sample":
                    sample = record["sample"]
                    assert isinstance(sample, Mapping)
                    raw_capture = sample.get("captured_ns")
                    if (
                        isinstance(raw_capture, bool)
                        or not isinstance(raw_capture, int)
                        or raw_capture <= 0
                    ):
                        raise RTX4090TwoHostCampaignError(
                            "attempt telemetry record capture is invalid"
                        )
                    capture = raw_capture
                    if last_capture is not None and capture <= last_capture:
                        raise RTX4090TwoHostCampaignError(
                            "attempt telemetry samples are not strictly increasing"
                        )
                    last_capture = capture
                    samples.append(dict(sample))
                    consecutive = 0
                else:
                    transient += int(outcome == "unavailable")
                    integrity += int(outcome == "integrity")
                    consecutive += 1
                    maximum = max(maximum, consecutive)
            record_root = self._attempt_telemetry_records_dir(request)
            try:
                actual = {
                    child.name for child in list_regular_file_children_nofollow(
                        record_root,
                        where="attempt telemetry record root",
                        allow_missing=True,
                    )
                }
            except ClusterTransportError as exc:
                raise RTX4090TwoHostCampaignError(
                    "attempt telemetry record root is unsafe"
                ) from exc
            expected_names = {
                f"{ordinal:020d}.json"
                for ordinal in range(int(head["record_count"]))
            }
            if actual != expected_names:
                raise RTX4090TwoHostCampaignError(
                    f"attempt telemetry record set differs for {command.work_id}"
                )
            host = _host_by_id(self.manifest, command.host_id)
            expected_host = host["expected"]
            assert isinstance(expected_host, Mapping)
            expected_gpu = expected_host["gpu"]
            assert isinstance(expected_gpu, Mapping)
            normalized_samples, _ = _normalize_telemetry(
                samples, required=False, expected_gpu=expected_gpu,
            )
            if samples != normalized_samples:
                raise RTX4090TwoHostCampaignError(
                    f"attempt telemetry records are noncanonical for {command.work_id}"
                )
            if (
                previous != head["head_record_sha256"]
                or transient != head["sampling_failure_count"]
                or integrity != head["sampling_integrity_failure_count"]
                or consecutive != head["consecutive_sampling_failure_count"]
                or maximum != head["maximum_consecutive_sampling_failure_count"]
                or last_capture != head["last_sample_captured_ns"]
                or head["pending_sample_ordinal"] is not None
            ):
                raise RTX4090TwoHostCampaignError(
                    f"attempt telemetry head/record chain differs for {command.work_id}"
                )
            return {
                "schema": schema,
                "samples": normalized_samples,
                "sampling_failure_count": transient,
                "sampling_integrity_failure_count": integrity,
                "consecutive_sampling_failure_count": consecutive,
                "maximum_consecutive_sampling_failure_count": maximum,
                "missing_prior_journal": bool(head["missing_prior_journal"]),
                "identity_sha256": str(head["identity_sha256"]),
            }
        expected_keys = (
            _LEGACY_ATTEMPT_TELEMETRY_KEYS
            if schema == LEGACY_ATTEMPT_TELEMETRY_SCHEMA
            else _LEGACY_ATTEMPT_TELEMETRY_V2_KEYS
        )
        if schema not in {
            LEGACY_ATTEMPT_TELEMETRY_SCHEMA,
            LEGACY_ATTEMPT_TELEMETRY_V2_SCHEMA,
        } or set(raw) != expected_keys:
            raise RTX4090TwoHostCampaignError(
                f"attempt telemetry fields differ for {command.work_id}"
            )
        body = dict(raw)
        identity = body.pop("identity_sha256", None)
        if identity != canonical_sha256(body):
            raise RTX4090TwoHostCampaignError(
                f"attempt telemetry digest differs for {command.work_id}"
            )
        exact = {
            "schema": schema,
            "campaign_identity_sha256": self.manifest["identity_sha256"],
            "work_id": command.work_id,
            "attempt_index": attempt_index,
            "request_sha256": request.request_sha256,
        }
        for key, expected_value in exact.items():
            if raw.get(key) != expected_value:
                raise RTX4090TwoHostCampaignError(
                    f"attempt telemetry {key} differs for {command.work_id}"
                )
        samples = raw["samples"]
        if not isinstance(samples, list):
            raise RTX4090TwoHostCampaignError(
                f"attempt telemetry samples are malformed for {command.work_id}"
            )
        host = _host_by_id(self.manifest, command.host_id)
        expected = host["expected"]
        assert isinstance(expected, Mapping)
        expected_gpu = expected["gpu"]
        assert isinstance(expected_gpu, Mapping)
        normalized, _ = _normalize_telemetry(
            samples, required=False, expected_gpu=expected_gpu,
        )
        if samples != normalized:
            raise RTX4090TwoHostCampaignError(
                f"attempt telemetry is noncanonical for {command.work_id}"
            )
        if schema == LEGACY_ATTEMPT_TELEMETRY_SCHEMA:
            # A v1 journal did not record sampling failures.  Its prefix is
            # therefore unknowable under the v2 acceptance contract.  The
            # durable legacy schema itself deterministically derives this
            # fail latch on every load; it is never rewritten or promoted.
            counters = {
                "sampling_failure_count": 0,
                "sampling_integrity_failure_count": 1,
                "consecutive_sampling_failure_count": 1,
                "maximum_consecutive_sampling_failure_count": 1,
                "missing_prior_journal": True,
            }
        else:
            counters = {
                key: raw[key]
                for key in (
                    "sampling_failure_count",
                    "sampling_integrity_failure_count",
                    "consecutive_sampling_failure_count",
                    "maximum_consecutive_sampling_failure_count",
                    "missing_prior_journal",
                )
            }
            self._validate_attempt_telemetry_counters(
                counters["sampling_failure_count"],
                counters["sampling_integrity_failure_count"],
                counters["consecutive_sampling_failure_count"],
                counters["maximum_consecutive_sampling_failure_count"],
                counters["missing_prior_journal"],
            )
        return {
            "schema": schema,
            "samples": normalized,
            **counters,
            "identity_sha256": str(identity),
        }

    def _append_attempt_telemetry(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        sample: object,
    ) -> dict[str, object]:
        return self._append_attempt_telemetry_record(
            command,
            request,
            attempt_index,
            outcome="sample",
            sample=sample,
            require_pending=True,
        )

    def _record_attempt_telemetry_failure(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        *,
        integrity_failure: bool,
        missing_prior_journal: bool = False,
    ) -> dict[str, object]:
        if missing_prior_journal and not integrity_failure:
            raise RTX4090TwoHostCampaignError(
                "loaded telemetry missing-prior-journal marker is invalid"
            )
        head = self._load_attempt_telemetry_head(
            command, request, attempt_index,
        )
        require_pending = head["pending_sample_ordinal"] is not None
        return self._append_attempt_telemetry_record(
            command,
            request,
            attempt_index,
            outcome="integrity" if integrity_failure else "unavailable",
            sample=None,
            require_pending=require_pending,
            missing_prior_journal=missing_prior_journal,
        )

    @staticmethod
    def _exact_transport_receipt(
        value: object, request: RunRequest,
    ) -> dict[str, object]:
        return _normalize_job_receipt(
            value, request, require_succeeded=False,
        )

    def _adopt_or_start(
        self,
        transport: CampaignTransport,
        request: RunRequest,
        *,
        before_start: Callable[[], None] | None = None,
        on_disposition: Callable[[str], None] | None = None,
        start_permitted: bool = True,
    ) -> tuple[dict[str, object], str]:
        """Return an exact job plus how this coordinator obtained it.

        ``adopted_after_start`` is distinct from an initially existing job:
        the caller ran ``before_start`` before the ambiguous start, so a
        request-bound journal created there is valid even when the start reply
        was lost and the exact job must be recovered by status.
        """

        try:
            existing = transport.status(request.job_id)
        except JobNotFoundError:
            if not start_permitted:
                raise RTX4090TwoHostCampaignError(
                    f"attempt {request.job_id} has unusable prior telemetry "
                    "but no durable transport state; refusing any retry"
                )
            if before_start is not None:
                before_start()
            try:
                started = transport.start(request)
            except JobConflictError:
                # Exact absence was observed, but this coordinator did not
                # create the now-claimed child.  Its prefix is therefore not
                # covered by our freshly published journal.
                try:
                    adopted = transport.status(request.job_id)
                except Exception as status_error:
                    raise RTX4090TwoHostCampaignError(
                        f"cannot safely adopt conflicting transport job "
                        f"{request.job_id}"
                    ) from status_error
                guard_validator = getattr(
                    transport, "validate_existing_guard", None,
                )
                if callable(guard_validator):
                    try:
                        guard_validator(request)
                    except Exception as exc:
                        raise RTX4090TwoHostCampaignError(
                            f"conflicting job {request.job_id} has no valid "
                            "launch guard"
                        ) from exc
                disposition = "existing_after_conflict"
                if on_disposition is not None:
                    on_disposition(disposition)
                return self._exact_transport_receipt(adopted, request), disposition
            except Exception:
                # Start can lose its response or reject an already claimed ID.
                # One exact status re-read is the only safe way to adopt it.
                try:
                    adopted = transport.status(request.job_id)
                except Exception as status_error:
                    raise RTX4090TwoHostCampaignError(
                        f"cannot safely adopt or start transport job "
                        f"{request.job_id}"
                    ) from status_error
                guard_validator = getattr(
                    transport, "validate_existing_guard", None,
                )
                if callable(guard_validator):
                    try:
                        guard_validator(request)
                    except Exception as exc:
                        raise RTX4090TwoHostCampaignError(
                            f"adopted job {request.job_id} has no valid "
                            "launch guard"
                        ) from exc
                disposition = "adopted_after_start"
                if on_disposition is not None:
                    on_disposition(disposition)
                return self._exact_transport_receipt(adopted, request), disposition
            disposition = "started"
            if on_disposition is not None:
                on_disposition(disposition)
            return self._exact_transport_receipt(started, request), disposition
        except Exception as exc:
            raise RTX4090TwoHostCampaignError(
                f"cannot determine whether transport job {request.job_id} "
                "exists; refusing start"
            ) from exc
        guard_validator = getattr(transport, "validate_existing_guard", None)
        if callable(guard_validator):
            try:
                guard_validator(request)
            except Exception as exc:
                raise RTX4090TwoHostCampaignError(
                    f"adopted job {request.job_id} has no valid launch guard"
                ) from exc
        disposition = "existing"
        if on_disposition is not None:
            on_disposition(disposition)
        return self._exact_transport_receipt(existing, request), disposition

    def _monitor_attempt(
        self,
        command: CampaignCommand,
        transport: CampaignTransport,
        request: RunRequest,
        attempt_index: int,
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        raw_cadence = getattr(
            transport, "cadence_seconds", self._telemetry_interval_seconds,
        )
        if isinstance(raw_cadence, bool):
            raise RTX4090TwoHostCampaignError(
                "transport telemetry cadence differs from campaign policy"
            )
        try:
            cadence_seconds = float(raw_cadence)
        except (TypeError, ValueError) as exc:
            raise RTX4090TwoHostCampaignError(
                "transport telemetry cadence is invalid"
            ) from exc
        if cadence_seconds != self._telemetry_interval_seconds:
            raise RTX4090TwoHostCampaignError(
                "transport telemetry cadence differs from campaign policy"
            )
        telemetry_path = self._attempt_telemetry_path(request)
        journal_existed_before_adoption = telemetry_path.exists()
        telemetry_state: dict[str, object] | None = None
        telemetry_head: dict[str, object] | None = None
        telemetry_load_error: Exception | None = None
        if command.gpu_bearing and journal_existed_before_adoption:
            try:
                raw_schema = _strict_json_mapping(
                    telemetry_path,
                    where=f"attempt telemetry for {command.work_id}",
                ).get("schema")
                if raw_schema == ATTEMPT_TELEMETRY_SCHEMA:
                    telemetry_head = self._recover_attempt_telemetry(
                        command, request, attempt_index,
                    )
                    telemetry_state = self._load_attempt_telemetry(
                        command, request, attempt_index,
                    )
                else:
                    telemetry_state = self._load_attempt_telemetry(
                        command, request, attempt_index,
                    )
            except Exception as exc:
                telemetry_load_error = exc
        pristine_current_journal = (
            telemetry_head is not None
            and telemetry_head["schema"] == ATTEMPT_TELEMETRY_SCHEMA
            and telemetry_head["record_count"] == 0
            and telemetry_head["pending_sample_ordinal"] is None
            and telemetry_head["start_phase"] == "prepared"
            and telemetry_head["sampling_failure_count"] == 0
            and telemetry_head["sampling_integrity_failure_count"] == 0
            and telemetry_head["consecutive_sampling_failure_count"] == 0
            and telemetry_head[
                "maximum_consecutive_sampling_failure_count"
            ] == 0
            and telemetry_head["missing_prior_journal"] is False
        )

        def before_start() -> None:
            if not telemetry_path.exists():
                self._initialize_attempt_telemetry(
                    command, request, attempt_index,
                )
            self._mark_attempt_start_invoked(
                command, request, attempt_index,
            )

        def on_disposition(value: str) -> None:
            if not command.gpu_bearing or telemetry_load_error is not None:
                return
            if not telemetry_path.exists():
                self._initialize_attempt_telemetry(
                    command,
                    request,
                    attempt_index,
                )
            raw = _strict_json_mapping(
                telemetry_path,
                where=f"attempt telemetry for {command.work_id}",
            )
            if raw.get("schema") != ATTEMPT_TELEMETRY_SCHEMA:
                return
            self._mark_attempt_start_claimed(
                command, request, attempt_index, value,
            )

        receipt, disposition = self._adopt_or_start(
            transport,
            request,
            before_start=(before_start if command.gpu_bearing else None),
            on_disposition=(on_disposition if command.gpu_bearing else None),
            start_permitted=(
                not command.gpu_bearing
                or not journal_existed_before_adoption
                or pristine_current_journal
            ),
        )
        if command.gpu_bearing:
            if telemetry_load_error is not None:
                # A claimed child must still reach durable terminal state
                # before a later attempt may start.  Malformed evidence is
                # never overwritten or treated as a clean prefix.
                while receipt["state"] == "running":
                    try:
                        current = transport.status(request.job_id)
                    except Exception as exc:
                        raise RTX4090TwoHostCampaignError(
                            f"status failed for running job {request.job_id}; "
                            "refusing a colliding retry"
                        ) from exc
                    receipt = self._exact_transport_receipt(current, request)
                    if receipt["state"] == "running":
                        self._sleep(cadence_seconds)
                raise _AttemptRetryRequired(
                    f"attempt {request.job_id} has malformed prior telemetry"
                ) from telemetry_load_error
            telemetry_state = self._load_attempt_telemetry(
                command, request, attempt_index,
            )
        else:
            telemetry_state = None
        while receipt["state"] == "running":
            if (
                command.gpu_bearing
                and telemetry_state is not None
                and telemetry_state["schema"] == ATTEMPT_TELEMETRY_SCHEMA
            ):
                try:
                    self._begin_attempt_telemetry_sample(
                        command, request, attempt_index,
                    )
                    sample = transport.sample_telemetry()
                    telemetry_state = self._append_attempt_telemetry(
                        command, request, attempt_index, sample,
                    )
                except TelemetryUnavailableError:
                    telemetry_state = self._record_attempt_telemetry_failure(
                        command, request, attempt_index,
                        integrity_failure=False,
                    )
                except Exception:
                    telemetry_state = self._record_attempt_telemetry_failure(
                        command, request, attempt_index,
                        integrity_failure=True,
                    )
            try:
                current = transport.status(request.job_id)
            except Exception as exc:
                raise RTX4090TwoHostCampaignError(
                    f"status failed for running job {request.job_id}; "
                    "refusing a colliding retry"
                ) from exc
            receipt = self._exact_transport_receipt(current, request)
            if receipt["state"] == "running":
                self._sleep(cadence_seconds)
        if command.gpu_bearing:
            telemetry_state = self._load_attempt_telemetry(
                command, request, attempt_index,
            )
        return receipt, telemetry_state

    def _execute(
        self, assignment: StageAssignment, *,
        dependency_receipt_sha256s: Sequence[str],
        precondition_receipt_sha256s: Sequence[str],
    ) -> tuple[StageAssignment, dict[str, object]]:
        self._require_current_manifest("execute")
        command = self.commands[assignment.work_id]
        transport = self.transports.get(assignment.host_id)
        if transport is None:
            raise RTX4090TwoHostCampaignError(
                f"no telemetry-capable transport for host {assignment.host_id!r}"
            )
        inspector = self.artifact_inspectors.get(assignment.host_id)
        if command.outputs and inspector is None:
            raise RTX4090TwoHostCampaignError(
                f"no artifact inspector for output-bearing host "
                f"{assignment.host_id!r}"
            )
        last_failure: BaseException | None = None
        for attempt_index in range(self._max_attempts):
            request = self._request(
                command,
                attempt_index=attempt_index,
                precondition_receipt_sha256s=precondition_receipt_sha256s,
            )
            try:
                job, attempt_telemetry = self._monitor_attempt(
                    command, transport, request, attempt_index,
                )
            except _AttemptRetryRequired as exc:
                last_failure = exc
                continue
            if job["state"] != "succeeded":
                last_failure = RTX4090TwoHostCampaignError(
                    f"transport job {request.job_id} ended in {job['state']}"
                )
                continue
            try:
                receipt = _stage_receipt(
                    self.manifest,
                    command,
                    request,
                    job,
                    attempt_telemetry,
                    dependency_receipt_sha256s,
                    precondition_receipt_sha256s,
                    attempt_index=attempt_index,
                    artifact_inspector=inspector,
                )
            except RTX4090TwoHostCampaignError as exc:
                last_failure = exc
                continue
            return assignment, receipt
        assert last_failure is not None
        raise RTX4090TwoHostCampaignError(
            f"assignment {assignment.work_id} exhausted "
            f"{self._max_attempts} deterministic attempt(s): {last_failure}"
        ) from last_failure

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

    def _validate_output_artifacts(
        self,
        receipt: Mapping[str, object],
        command: CampaignCommand,
        host: Mapping[str, object],
    ) -> None:
        stored = receipt.get("output_artifacts")
        if not isinstance(stored, list) or len(stored) != len(command.outputs):
            raise RTX4090TwoHostCampaignError(
                f"execution output ledger is malformed for {command.work_id}"
            )
        normalized: list[dict[str, object]] = []
        for index, (raw_row, container_path) in enumerate(
            zip(stored, command.outputs, strict=True)
        ):
            if not isinstance(raw_row, Mapping) or set(raw_row) != _OUTPUT_ARTIFACT_KEYS:
                raise RTX4090TwoHostCampaignError(
                    f"execution output ledger row {index} differs for "
                    f"{command.work_id}"
                )
            row = dict(raw_row)
            host_path, _ = _host_output_path(host, container_path)
            expected_kind = _expected_output_kind(container_path)
            if (
                row.get("container_path") != container_path
                or row.get("host_path") != host_path
                or row.get("expected_kind") != expected_kind
            ):
                raise RTX4090TwoHostCampaignError(
                    f"execution output path binding differs for {command.work_id}"
                )
            manifest = _normalize_tree_manifest(
                row.get("manifest"),
                where=f"stored output {container_path}",
            )
            if (
                manifest.root_kind != expected_kind
                or row.get("manifest") != manifest.to_payload()
                or row.get("manifest_sha256") != manifest.identity_sha256
            ):
                raise RTX4090TwoHostCampaignError(
                    f"execution output manifest differs for {command.work_id}"
                )
            normalized.append({
                "container_path": container_path,
                "host_path": host_path,
                "expected_kind": expected_kind,
                "manifest": manifest.to_payload(),
                "manifest_sha256": manifest.identity_sha256,
            })
        if stored != normalized:
            raise RTX4090TwoHostCampaignError(
                f"execution output ledger is noncanonical for {command.work_id}"
            )
        current = _inspect_command_outputs(
            command,
            host,
            self.artifact_inspectors.get(command.host_id),
        )
        if current != normalized:
            raise RTX4090TwoHostCampaignError(
                f"declared output content drifted for {command.work_id}"
            )

    def _validate_receipt(
        self,
        raw: object,
        assignment: StageAssignment,
        state: Mapping[str, object],
        precondition_receipt_sha256s: Sequence[str],
    ) -> dict[str, object]:
        if not isinstance(raw, Mapping):
            raise RTX4090TwoHostCampaignError(
                f"execution receipt fields differ for {assignment.work_id}"
            )
        schema = raw.get("schema")
        if schema == LEGACY_EXECUTION_RECEIPT_SCHEMA:
            expected_keys = _LEGACY_EXECUTION_RECEIPT_KEYS
        elif schema == EXECUTION_RECEIPT_SCHEMA:
            expected_keys = _EXECUTION_RECEIPT_KEYS
        else:
            raise RTX4090TwoHostCampaignError(
                f"execution receipt schema differs for {assignment.work_id}"
            )
        manifest_schema = self.manifest["schema"]
        if (
            manifest_schema == LEGACY_CAMPAIGN_MANIFEST_SCHEMA
            and schema != LEGACY_EXECUTION_RECEIPT_SCHEMA
        ) or (
            manifest_schema == CAMPAIGN_MANIFEST_SCHEMA
            and schema != EXECUTION_RECEIPT_SCHEMA
        ):
            raise RTX4090TwoHostCampaignError(
                "execution receipt schema is incompatible with campaign "
                f"manifest for {assignment.work_id}"
            )
        if set(raw) != expected_keys:
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
            "schema": schema,
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
            "precondition_receipt_sha256s": list(
                self._normalize_precondition_sha256s(
                    precondition_receipt_sha256s,
                )
            ),
        }
        for key, expected in exact.items():
            if receipt.get(key) != expected:
                raise RTX4090TwoHostCampaignError(
                    f"execution receipt {key} differs for {assignment.work_id}"
                )
        attempt_index = receipt.get("attempt_index")
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or not 0 <= attempt_index < self._max_attempts
        ):
            raise RTX4090TwoHostCampaignError(
                f"execution receipt attempt differs for {assignment.work_id}"
            )
        request = self._request(
            command,
            attempt_index=attempt_index,
            precondition_receipt_sha256s=precondition_receipt_sha256s,
        )
        if request.request_sha256 != receipt.get("request_sha256"):
            raise RTX4090TwoHostCampaignError(
                f"execution receipt request differs for {assignment.work_id}"
            )
        normalized_job = _normalize_job_receipt(receipt["job"], request)
        if receipt["job"] != normalized_job:
            raise RTX4090TwoHostCampaignError(
                f"execution job receipt is noncanonical for {assignment.work_id}"
            )
        activity_requirement = _gpu_activity_requirement(
            self.manifest,
            command,
            normalized_job,
            attempt_index=attempt_index,
            legacy_retry_waiver=(
                schema == LEGACY_EXECUTION_RECEIPT_SCHEMA
            ),
        )
        if receipt.get("gpu_activity_requirement") != activity_requirement:
            raise RTX4090TwoHostCampaignError(
                "execution GPU activity requirement differs for "
                f"{assignment.work_id}"
            )
        stored_telemetry = receipt["telemetry"]
        if not isinstance(stored_telemetry, list):
            raise RTX4090TwoHostCampaignError(
                f"execution telemetry is malformed for {assignment.work_id}"
            )
        stored_summary = receipt.get("telemetry_summary")
        if not isinstance(stored_summary, Mapping):
            raise RTX4090TwoHostCampaignError(
                f"execution telemetry summary is malformed for "
                f"{assignment.work_id}"
            )
        if schema == LEGACY_EXECUTION_RECEIPT_SCHEMA:
            expected = host["expected"]
            assert isinstance(expected, Mapping)
            expected_gpu = expected["gpu"]
            assert isinstance(expected_gpu, Mapping)
            samples, summary = _normalize_telemetry(
                stored_telemetry,
                required=(
                    activity_requirement == "positive_utilization_required"
                ),
                expected_gpu=expected_gpu,
            )
        else:
            attempt_telemetry = (
                self._load_attempt_telemetry(
                    command,
                    request,
                    attempt_index,
                    create_if_missing=False,
                )
                if command.gpu_bearing else None
            )
            samples, summary, journal_identity = (
                _normalize_attempt_telemetry_evidence(
                    self.manifest,
                    command,
                    normalized_job,
                    attempt_telemetry,
                    activity_requirement=activity_requirement,
                    allow_legacy_v2=True,
                )
            )
            if (
                receipt.get("attempt_telemetry_identity_sha256")
                != journal_identity
            ):
                raise RTX4090TwoHostCampaignError(
                    "execution attempt telemetry identity differs for "
                    f"{assignment.work_id}"
                )
        if (
            stored_telemetry != samples
            or receipt.get("telemetry_sha256") != canonical_sha256(samples)
            or stored_summary != summary
        ):
            raise RTX4090TwoHostCampaignError(
                f"execution telemetry differs for {assignment.work_id}"
            )
        self._validate_output_artifacts(receipt, command, host)
        return receipt

    def _read_receipt(
        self, assignment: StageAssignment, state: Mapping[str, object],
        precondition_receipt_sha256s: Sequence[str],
    ) -> dict[str, object]:
        path = self.state_dir / RECEIPT_DIR / f"{assignment.work_id}.json"
        raw = _strict_json_mapping(
            path, where=f"execution receipt {assignment.work_id}",
        )
        return self._validate_receipt(
            raw, assignment, state, precondition_receipt_sha256s,
        )

    def advance(
        self, state: Mapping[str, object], *, resume: bool,
    ) -> dict[str, object]:
        self._require_current_manifest("advance")
        ready = next_ready_assignments(self.manifest, state)
        if not ready:
            return dict(state)
        stage = ready[0].stage
        if any(assignment.stage != stage for assignment in ready):
            raise RTX4090TwoHostCampaignError(
                "campaign frontier unexpectedly spans multiple stages"
            )
        preconditions = self._stage_preconditions(stage, verify_only=False)
        current = dict(state)
        pending: list[StageAssignment] = []
        for assignment in ready:
            receipt_path = (
                self.state_dir / RECEIPT_DIR / f"{assignment.work_id}.json"
            )
            if not receipt_path.exists():
                pending.append(assignment)
                continue
            receipt = self._read_receipt(
                assignment, current, preconditions,
            )
            current = complete_assignment(
                self.manifest,
                current,
                assignment,
                str(receipt["identity_sha256"]),
            )
            _replace_state(self.state_path, current)
        if not pending:
            return current
        failures: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futures = {
                pool.submit(
                    self._execute,
                    assignment,
                    dependency_receipt_sha256s=self._dependency_receipts(
                        current, assignment.stage,
                    ),
                    precondition_receipt_sha256s=preconditions,
                ): assignment
                for assignment in pending
            }
            for future in as_completed(futures):
                try:
                    assignment, receipt = future.result()
                except BaseException as exc:  # retain successful peer progress
                    failures.append(exc)
                    continue
                receipt_path = (
                    self.state_dir / RECEIPT_DIR /
                    f"{assignment.work_id}.json"
                )
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
        self._require_current_manifest("run or resume")
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
            manifest_bytes = read_regular_file_nofollow(
                manifest_path, where="stored campaign manifest",
            )
            stored_manifest = parse_campaign_manifest(
                manifest_bytes.decode("utf-8")
            )
        except (
            ClusterTransportError, UnicodeDecodeError, ValueError,
        ) as exc:
            raise RTX4090TwoHostCampaignError(
                "stored campaign manifest/plan is unreadable"
            ) from exc
        if stored_manifest != self.manifest:
            raise RTX4090TwoHostCampaignError("stored campaign manifest differs")
        if manifest_bytes != _json_bytes(self.manifest):
            raise RTX4090TwoHostCampaignError(
                "stored campaign manifest bytes are noncanonical"
            )
        raw_plan = _strict_json_mapping(plan_path, where="stored command plan")
        expected_plan = validate_command_plan(raw_plan, self.manifest)
        assert raw_plan == expected_plan
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
        try:
            paths = list(list_regular_file_children_nofollow(
                receipt_root,
                where="execution receipt directory",
                allow_missing=True,
            ))
        except ClusterTransportError as exc:
            raise RTX4090TwoHostCampaignError(
                "execution receipt directory is unsafe"
            ) from exc
        if {path.stem for path in paths} != set(completion_by_work_id):
            raise RTX4090TwoHostCampaignError(
                "execution receipt files differ from durable state completions"
            )
        preconditions_by_stage: dict[str, tuple[str, ...]] = {}
        for path in paths:
            work_id = path.stem
            assignment = assignment_by_work_id.get(work_id)
            if assignment is None:
                raise RTX4090TwoHostCampaignError(
                    f"execution receipt is outside the fixed DAG: {path}"
                )
            preconditions = preconditions_by_stage.get(assignment.stage)
            if preconditions is None:
                preconditions = self._stage_preconditions(
                    assignment.stage, verify_only=True,
                )
                preconditions_by_stage[assignment.stage] = preconditions
            receipt = self._read_receipt(
                assignment, state, preconditions,
            )
            completion = completion_by_work_id[work_id]
            if completion.get("receipt_sha256") != receipt["identity_sha256"]:
                raise RTX4090TwoHostCampaignError(
                    f"execution receipt is not state-bound: {path}"
                )
        return self.status()


def _load_manifest(path: str | Path) -> dict[str, object]:
    try:
        payload = read_regular_file_nofollow(
            Path(path), where="campaign manifest",
        )
        return parse_campaign_manifest(payload.decode("utf-8"))
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
ArtifactInspectorFactory = Callable[
    [Mapping[str, object]], Mapping[str, ArtifactInspector]
]


def main(
    argv: Sequence[str] | None = None,
    *,
    transport_factory: TransportFactory | None = None,
    artifact_inspector_factory: ArtifactInspectorFactory | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if args.command == "worker-preflight":
        print(json.dumps(_worker_preflight(args), sort_keys=True))
        return 0
    manifest = _load_manifest(args.manifest)
    if args.command == "plan":
        if manifest["schema"] == LEGACY_CAMPAIGN_MANIFEST_SCHEMA:
            raise RTX4090TwoHostCampaignError(
                "legacy campaign manifests are read-only; cannot write plan"
            )
        _write_exact_no_clobber(Path(args.output), build_command_plan(manifest))
        print(args.output)
        return 0
    if transport_factory is None and artifact_inspector_factory is None:
        if args.command == "status":
            result = CampaignCoordinator(manifest, args.state_dir).status()
        else:
            from prismaquant.rtx4090_two_host_application import (
                build_live_campaign_application,
            )

            try:
                application = build_live_campaign_application(
                    manifest,
                    args.state_dir,
                    initialize=args.command in {"run", "resume"},
                )
                if args.command == "run":
                    result = application.run_to_completion(resume=False)
                elif args.command == "resume":
                    result = application.run_to_completion(resume=True)
                else:
                    result = application.verify()
            except RTX4090TwoHostCampaignError:
                raise
            except Exception as exc:
                raise RTX4090TwoHostCampaignError(
                    f"live campaign application failed: {exc}"
                ) from exc
        print(json.dumps(result, sort_keys=True))
        return 0
    transports = transport_factory(manifest) if transport_factory is not None else {}
    artifact_inspectors = (
        artifact_inspector_factory(manifest)
        if artifact_inspector_factory is not None else {}
    )
    coordinator = CampaignCoordinator(
        manifest,
        args.state_dir,
        transports=transports,
        artifact_inspectors=artifact_inspectors,
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
