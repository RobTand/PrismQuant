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
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence

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
    TreeManifest,
    build_tree_manifest,
    canonical_json_bytes,
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
ATTEMPT_TELEMETRY_DIR = "attempt-telemetry"
LOCK_FILE = "campaign.lock"
ATTEMPT_TELEMETRY_SCHEMA = (
    "prismaquant.rtx4090_two_host_campaign.attempt_telemetry.v1"
)
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
})
_RESUMABLE_FLAG_STAGES = frozenset({"measure_burn", "allocate"})
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
}
_CONTAINER_CLEANUP_GRACE_SECONDS = 600.0
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
_OUTPUT_ARTIFACT_KEYS = frozenset({
    "container_path",
    "host_path",
    "expected_kind",
    "manifest",
    "manifest_sha256",
})
_ATTEMPT_TELEMETRY_KEYS = frozenset({
    "schema",
    "campaign_identity_sha256",
    "work_id",
    "attempt_index",
    "request_sha256",
    "samples",
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
    return "directory" if name in {"act", "activation_cache"} else "file"


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
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except RTX4090TwoHostCampaignError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
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


def _stage_receipt(
    manifest: Mapping[str, object], command: CampaignCommand,
    request: RunRequest, job_receipt: object, telemetry: Sequence[object],
    dependency_receipt_sha256s: Sequence[str],
    precondition_receipt_sha256s: Sequence[str], *, attempt_index: int,
    artifact_inspector: ArtifactInspector | None,
) -> dict[str, object]:
    job = _normalize_job_receipt(job_receipt, request)
    host = _host_by_id(manifest, command.host_id)
    expected = host["expected"]
    assert isinstance(expected, Mapping)
    expected_gpu = expected["gpu"]
    assert isinstance(expected_gpu, Mapping)
    activity_requirement = _gpu_activity_requirement(
        manifest, command, job,
    )
    samples, telemetry_summary = _normalize_telemetry(
        telemetry,
        required=activity_requirement == "positive_utilization_required",
        expected_gpu=expected_gpu,
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
        raw = _strict_json_mapping(
            self.state_path, where="campaign state",
        )
        return validate_campaign_state(raw, self.manifest)

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

    def _attempt_telemetry_payload(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        samples: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": ATTEMPT_TELEMETRY_SCHEMA,
            "campaign_identity_sha256": self.manifest["identity_sha256"],
            "work_id": command.work_id,
            "attempt_index": attempt_index,
            "request_sha256": request.request_sha256,
            "samples": [dict(sample) for sample in samples],
        }
        return {**body, "identity_sha256": canonical_sha256(body)}

    def _load_attempt_telemetry(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
    ) -> list[dict[str, object]]:
        path = self._attempt_telemetry_path(request)
        if not path.exists():
            empty = self._attempt_telemetry_payload(
                command, request, attempt_index, (),
            )
            _write_exact_no_clobber(path, empty)
        raw = _strict_json_mapping(
            path, where=f"attempt telemetry for {command.work_id}",
        )
        if set(raw) != _ATTEMPT_TELEMETRY_KEYS:
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
            "schema": ATTEMPT_TELEMETRY_SCHEMA,
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
        return normalized

    def _append_attempt_telemetry(
        self,
        command: CampaignCommand,
        request: RunRequest,
        attempt_index: int,
        sample: object,
    ) -> list[dict[str, object]]:
        samples = self._load_attempt_telemetry(
            command, request, attempt_index,
        )
        host = _host_by_id(self.manifest, command.host_id)
        expected = host["expected"]
        assert isinstance(expected, Mapping)
        expected_gpu = expected["gpu"]
        assert isinstance(expected_gpu, Mapping)
        normalized, _ = _normalize_telemetry(
            [*samples, sample], required=False, expected_gpu=expected_gpu,
        )
        payload = self._attempt_telemetry_payload(
            command, request, attempt_index, normalized,
        )
        _replace_state(self._attempt_telemetry_path(request), payload)
        return normalized

    @staticmethod
    def _exact_transport_receipt(
        value: object, request: RunRequest,
    ) -> dict[str, object]:
        return _normalize_job_receipt(
            value, request, require_succeeded=False,
        )

    def _adopt_or_start(
        self, transport: CampaignTransport, request: RunRequest,
    ) -> dict[str, object]:
        try:
            existing = transport.status(request.job_id)
        except Exception:
            try:
                started = transport.start(request)
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
                return self._exact_transport_receipt(adopted, request)
            return self._exact_transport_receipt(started, request)
        guard_validator = getattr(transport, "validate_existing_guard", None)
        if callable(guard_validator):
            try:
                guard_validator(request)
            except Exception as exc:
                raise RTX4090TwoHostCampaignError(
                    f"adopted job {request.job_id} has no valid launch guard"
                ) from exc
        return self._exact_transport_receipt(existing, request)

    def _monitor_attempt(
        self,
        command: CampaignCommand,
        transport: CampaignTransport,
        request: RunRequest,
        attempt_index: int,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
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
        samples = (
            self._load_attempt_telemetry(command, request, attempt_index)
            if command.gpu_bearing else []
        )
        receipt = self._adopt_or_start(transport, request)
        while receipt["state"] == "running":
            if command.gpu_bearing:
                try:
                    sample = transport.sample_telemetry()
                except Exception as exc:
                    raise RTX4090TwoHostCampaignError(
                        f"live telemetry failed for {request.job_id}"
                    ) from exc
                samples = self._append_attempt_telemetry(
                    command, request, attempt_index, sample,
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
        return receipt, samples

    def _execute(
        self, assignment: StageAssignment, *,
        dependency_receipt_sha256s: Sequence[str],
        precondition_receipt_sha256s: Sequence[str],
    ) -> tuple[StageAssignment, dict[str, object]]:
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
            job, samples = self._monitor_attempt(
                command, transport, request, attempt_index,
            )
            if job["state"] != "succeeded":
                last_failure = RTX4090TwoHostCampaignError(
                    f"transport job {request.job_id} ended in {job['state']}"
                )
                continue
            try:
                receipt = _stage_receipt(
                    self.manifest, command, request, job, samples,
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
            self.manifest, command, normalized_job,
        )
        if receipt.get("gpu_activity_requirement") != activity_requirement:
            raise RTX4090TwoHostCampaignError(
                "execution GPU activity requirement differs for "
                f"{assignment.work_id}"
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
            required=activity_requirement == "positive_utilization_required",
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
            stored_manifest = parse_campaign_manifest(
                manifest_bytes.decode("utf-8")
            )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
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
        paths = sorted(receipt_root.glob("*.json")) if receipt_root.exists() else []
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
