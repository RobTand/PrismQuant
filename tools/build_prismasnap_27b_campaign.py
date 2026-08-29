#!/usr/bin/env python3
"""Build the pinned two-Spark Qwen3.8-27B PrismaSnap campaign.

This is a release-specific, versioned campaign compiler.  ``build`` prepares
and verifies immutable inputs on both hosts and writes a sealed v2 manifest;
it never starts the campaign.  Runtime-only subcommands are used by sealed
stages for host preflight and content-verified transfers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CLUSTER_CAMPAIGN_PATH = _REPOSITORY_ROOT / "prismaquant" / "cluster_campaign.py"
_CLUSTER_SPEC = importlib.util.spec_from_file_location(
    "_prismaquant_cluster_campaign_standalone", _CLUSTER_CAMPAIGN_PATH
)
if _CLUSTER_SPEC is None or _CLUSTER_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load cluster campaign module: {_CLUSTER_CAMPAIGN_PATH}")
_cluster_campaign = importlib.util.module_from_spec(_CLUSTER_SPEC)
_CLUSTER_SPEC.loader.exec_module(_cluster_campaign)
CAMPAIGN_MANIFEST_SCHEMA_V2 = _cluster_campaign.CAMPAIGN_MANIFEST_SCHEMA_V2
canonical_sha256 = _cluster_campaign.canonical_sha256
seal_campaign_manifest_v2 = _cluster_campaign.seal_campaign_manifest_v2
sealed_stage_receipt_sha256 = _cluster_campaign.sealed_stage_receipt_sha256
# Runtime subcommands executed in the pinned container import the production
# package only after argument dispatch.  Merely exposing its root on sys.path
# keeps host-only hash/transfer commands Torch-free.
sys.path.insert(0, str(_REPOSITORY_ROOT))


CAMPAIGN_BUILDER_SCHEMA = "prismaquant.prismasnap.qwen38_27b_campaign.v1"
CAMPAIGN_INPUTS_SCHEMA = "prismaquant.prismasnap.campaign_inputs.v1"
TRANSFER_RECEIPT_SCHEMA = "prismaquant.prismasnap.transfer_receipt.v1"
CAMPAIGN_ID = "qwen38-27b-prismasnap-20gb-20260825"
RUN_ROOT = Path("/home/rob/dq-runs/prismasnap-qwen38-27b-20gb-20260825")
CODE_ROOT = RUN_ROOT / "inputs" / "code"
SOURCE_ROOT = Path(
    "/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/source-text-mtp"
)
HF_ALIAS = Path(
    "/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/"
    "snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
ORIGINAL_PROBE_SOURCE = Path(
    "/home/rob/dq-runs/qwen38-27b-arm-b/artifacts/probe.pkl"
)
SOURCE_IDENTITY_SOURCE = Path(
    "/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/artifacts/"
    "source_model_identity.json"
)
ORIGINAL_PROBE_SHA256 = (
    "dbe7ee001e857a284c66f13831aae5809bbe5551f147cfbf91f6391b38097169"
)
SOURCE_IDENTITY_SHA256 = (
    "5f85249f4d2855d1948674594cfee927166937a3709bf165422aa878a702a7dc"
)
SOURCE_CONFIG_SHA256 = (
    "e655692e823db38aa68bd9f61e23ae423669519665e1b90f485f8d363a7a1d7a"
)
SOURCE_INDEX_SHA256 = (
    "fb525f9d6e93cc5172190b73732a526380adf1875698834ef110a67c79a647d7"
)
SOURCE_PORTABLE_SHA256 = (
    "6a7d9f66d85062bdb9990b556950a197044776142ccabb43e30b0f7756d846cb"
)
SOURCE_SHARD_BYTES = 54_641_508_824
ROOTFS_LAYERS_SHA256 = (
    "72c4e43ca2bb57d2e1c518be00122ef45f31baedc11ba60c2d8684cc9e700fca"
)
LOCAL_IMAGE_ID = (
    "sha256:e31deb853c5baa38ba09db34cffef8c8e91cdab64596a2bb0b7d40bfd8726a54"
)
REMOTE_IMAGE_ID = (
    "sha256:861db7c61c630924708d3566881f55cf1695352071cc630a8d280c12c8ddc3f0"
)
LOCAL_GPU_UUID = "GPU-e76c7efc-c157-b1f4-1348-83e4eb5092f4"
REMOTE_GPU_UUID = "GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705"
GPU_NAME = "NVIDIA GB10"
GPU_COMPUTE_CAPABILITY = "12.1"
REMOTE_ADDRESS = "10.100.96.2"
REMOTE_USER = "rob"
SSH_KEY = Path("/home/rob/.ssh/id_ed25519_nvsync_cluster_assistant")
KNOWN_HOSTS = Path("/home/rob/.ssh/known_hosts")
SSH_SHIM = RUN_ROOT / "inputs" / "ssh-sparklina"
LOCAL_MIN_FREE_BYTES = 270_000_000_000
REMOTE_MIN_FREE_BYTES = 135_000_000_000
CODE_SUBTREES = ("prismaquant", "tools")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

SPARKY_SHARDS = tuple(
    f"model-{index:05d}-of-00018.safetensors" for index in range(1, 10)
)
SPARKLINA_SHARDS = tuple(
    f"model-{index:05d}-of-00018.safetensors" for index in range(10, 19)
)


class CampaignBuildError(RuntimeError):
    """A pinned campaign input or generated contract is unsafe."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CampaignBuildError("value is not finite canonical JSON") from exc


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CampaignBuildError(f"expected one regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise CampaignBuildError(f"duplicate JSON member {key!r} in {path}")
            result[key] = value
        return result

    def reject(value: str) -> object:
        raise CampaignBuildError(f"non-JSON constant {value!r} in {path}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject,
        )
    except CampaignBuildError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignBuildError(f"cannot load exact JSON {path}") from exc
    if not isinstance(value, dict):
        raise CampaignBuildError(f"JSON root must be an object: {path}")
    return value


def _atomic_write_or_verify(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise CampaignBuildError(f"existing pinned file differs: {path}")
        if stat.S_IMODE(path.stat().st_mode) != mode:
            raise CampaignBuildError(f"existing pinned file mode differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_or_verify(source: Path, destination: Path, expected_sha256: str) -> None:
    observed = _sha256_file(source)
    if observed != expected_sha256:
        raise CampaignBuildError(
            f"pinned source hash differs for {source}: {observed}"
        )
    _atomic_write_or_verify(destination, source.read_bytes(), mode=0o444)


def _tree_manifest(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise CampaignBuildError(f"tree root must be one real directory: {root}")
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise CampaignBuildError(f"tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CampaignBuildError(f"tree contains a non-regular entry: {path}")
        rows.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not rows:
        raise CampaignBuildError(f"tree is empty: {root}")
    unsigned = {"files": rows}
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _code_manifest(code_root: Path) -> dict[str, object]:
    subtrees = {name: _tree_manifest(code_root / name) for name in CODE_SUBTREES}
    return {"subtrees": subtrees, "sha256": canonical_sha256(subtrees)}


def _rootfs_layers_sha256(layers: Sequence[str]) -> str:
    encoded = _canonical_bytes(list(layers)) + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def _docker_inspect(image_id: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["/usr/bin/docker", "image", "inspect", image_id],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise CampaignBuildError("docker inspect returned an unexpected document")
    return value[0]


def _verify_image(image_id: str) -> None:
    image = _docker_inspect(image_id)
    if image.get("Id") != image_id:
        raise CampaignBuildError(f"Docker image ID differs: {image.get('Id')!r}")
    rootfs = image.get("RootFS")
    layers = rootfs.get("Layers") if isinstance(rootfs, Mapping) else None
    if not isinstance(layers, list) or not all(isinstance(row, str) for row in layers):
        raise CampaignBuildError("Docker image has no exact RootFS layer list")
    observed = _rootfs_layers_sha256(layers)
    if observed != ROOTFS_LAYERS_SHA256:
        raise CampaignBuildError(f"Docker RootFS layer identity differs: {observed}")


def _ssh_base() -> list[str]:
    return [
        str(SSH_SHIM),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-o",
        "ConnectTimeout=20",
        "-p",
        "22",
        "--",
        f"{REMOTE_USER}@{REMOTE_ADDRESS}",
    ]


def _rsync_argv(
    *, direction: str, source: Path, destination: Path
) -> list[str]:
    if direction not in {"push", "fetch"}:
        raise CampaignBuildError("transfer direction must be push or fetch")
    common = [
        "/usr/bin/rsync",
        "--archive",
        "--checksum",
        "--partial",
        "--protect-args",
        "--mkpath",
        "--rsync-path=/usr/bin/rsync",
        f"--rsh={SSH_SHIM}",
    ]
    remote = f"{REMOTE_USER}@{REMOTE_ADDRESS}"
    source_arg = str(source) + "/"
    destination_arg = str(destination) + "/"
    if direction == "push":
        return [*common, source_arg, f"{remote}:{destination_arg}"]
    return [*common, f"{remote}:{source_arg}", destination_arg]


def _remote_tree_manifest(path: Path) -> dict[str, object]:
    command = [
        *_ssh_base(),
        "/usr/bin/python3",
        "-B",
        str(CODE_ROOT / "tools" / Path(__file__).name),
        "hash-tree",
        "--path",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise CampaignBuildError("remote tree manifest is not an object")
    return value


def transfer_tree(
    *, direction: str, source: Path, destination: Path, receipt: Path
) -> dict[str, object]:
    local_path = source if direction == "push" else destination
    remote_path = destination if direction == "push" else source
    before_local = _tree_manifest(local_path) if direction == "push" else None
    before_remote = _remote_tree_manifest(remote_path) if direction == "fetch" else None
    subprocess.run(
        _rsync_argv(direction=direction, source=source, destination=destination),
        check=True,
    )
    after_local = _tree_manifest(local_path)
    after_remote = _remote_tree_manifest(remote_path)
    expected = before_local if direction == "push" else before_remote
    if expected != after_local or expected != after_remote:
        raise CampaignBuildError("rsync source/destination content manifests differ")
    unsigned: dict[str, object] = {
        "schema": TRANSFER_RECEIPT_SCHEMA,
        "direction": direction,
        "source": str(source),
        "destination": str(destination),
        "remote": f"{REMOTE_USER}@{REMOTE_ADDRESS}",
        "content": expected,
    }
    payload = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    _atomic_write_or_verify(receipt, _pretty_bytes(payload), mode=0o444)
    return payload


def _ssh_shim_bytes() -> bytes:
    return (
        "#!/usr/bin/python3\n"
        "import os, sys\n"
        f"os.execv('/usr/bin/ssh', ['/usr/bin/ssh', '-F', '/dev/null', "
        f"'-i', '{SSH_KEY}', '-o', 'BatchMode=yes', '-o', "
        f"'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile={KNOWN_HOSTS}', "
        "'-o', 'ConnectTimeout=20', *sys.argv[1:]])\n"
    ).encode("utf-8")


def _docker_prefix(image_id: str, commit: str, *, gpu: bool) -> list[str]:
    argv = [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--user",
        "1000:1000",
    ]
    if gpu:
        argv.extend(["--gpus", "all"])
    argv.extend(
        [
            "--env",
            f"PRISMAQUANT_PRODUCER_GIT_COMMIT={commit}",
            "--env",
            f"PRISMAQUANT_CONTAINER_ROOTFS_SHA256={ROOTFS_LAYERS_SHA256}",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONHASHSEED=0",
            "--env",
            "HOME=/tmp/prismasnap-home",
            "--env",
            "XDG_CACHE_HOME=/tmp/prismasnap-cache",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=4294967296",
            "--mount",
            f"type=bind,src={CODE_ROOT},dst=/campaign/code,readonly",
            "--mount",
            f"type=bind,src={SOURCE_ROOT},dst={HF_ALIAS},readonly",
            "--mount",
            f"type=bind,src={RUN_ROOT},dst={RUN_ROOT}",
            "--workdir",
            "/campaign/code",
            "--entrypoint",
            "/usr/bin/python3",
            image_id,
            "-B",
        ]
    )
    return argv


def _prismasnap_argv(
    image_id: str, commit: str, command: Sequence[str], *, gpu: bool
) -> list[str]:
    return [
        *_docker_prefix(image_id, commit, gpu=gpu),
        "/campaign/code/tools/prismasnap.py",
        *command,
    ]


def _host_config(host_id: str) -> tuple[str, str, int]:
    if host_id == "sparky":
        return LOCAL_IMAGE_ID, LOCAL_GPU_UUID, LOCAL_MIN_FREE_BYTES
    if host_id == "sparklina":
        return REMOTE_IMAGE_ID, REMOTE_GPU_UUID, REMOTE_MIN_FREE_BYTES
    raise CampaignBuildError(f"unsupported campaign host: {host_id}")


def _verify_gpu(expected_uuid: str) -> None:
    query = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=uuid,name,compute_cap,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip().splitlines()
    if len(query) != 1:
        raise CampaignBuildError(f"expected exactly one GPU, observed {len(query)}")
    fields = [item.strip() for item in query[0].split(",")]
    if len(fields) != 4:
        raise CampaignBuildError("nvidia-smi GPU identity row is malformed")
    if fields[:3] != [expected_uuid, GPU_NAME, GPU_COMPUTE_CAPABILITY]:
        raise CampaignBuildError(f"GPU identity differs: {fields[:3]!r}")
    try:
        utilization = int(fields[3])
    except ValueError as exc:
        raise CampaignBuildError("GPU utilization is unavailable") from exc
    if utilization > 5:
        raise CampaignBuildError(f"GPU is not idle: utilization={utilization}%")
    processes = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if processes:
        raise CampaignBuildError(f"GPU has active compute processes: {processes}")


def _verify_source(*, full_content: bool) -> None:
    identity_path = RUN_ROOT / "inputs" / "source_model_identity.json"
    if _sha256_file(identity_path) != SOURCE_IDENTITY_SHA256:
        raise CampaignBuildError("copied source identity hash differs")
    if _sha256_file(SOURCE_ROOT / "config.json") != SOURCE_CONFIG_SHA256:
        raise CampaignBuildError("source config hash differs")
    if _sha256_file(SOURCE_ROOT / "model.safetensors.index.json") != SOURCE_INDEX_SHA256:
        raise CampaignBuildError("source safetensors index hash differs")
    identity = _load_json(identity_path)
    if identity.get("schema") != "prismaquant.streamed_model.identity.v1":
        raise CampaignBuildError("source identity schema differs")
    shards = identity.get("shards")
    if not isinstance(shards, list) or len(shards) != 18:
        raise CampaignBuildError("source identity must contain exactly 18 shards")
    expected_names = [*SPARKY_SHARDS, *SPARKLINA_SHARDS]
    observed_names: list[str] = []
    total = 0
    for row in shards:
        if not isinstance(row, Mapping):
            raise CampaignBuildError("source shard identity row is malformed")
        name = Path(str(row.get("path"))).name
        size = row.get("size")
        digest = row.get("sha256")
        if type(size) is not int or not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise CampaignBuildError("source shard identity fields are malformed")
        path = SOURCE_ROOT / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise CampaignBuildError(f"source shard stat differs: {path}")
        if full_content and _sha256_file(path) != digest:
            raise CampaignBuildError(f"source shard content differs: {path}")
        observed_names.append(name)
        total += size
    if observed_names != expected_names or total != SOURCE_SHARD_BYTES:
        raise CampaignBuildError("source shard set/byte count differs")


def _verify_inputs_identity() -> dict[str, Any]:
    path = RUN_ROOT / "inputs" / "campaign_inputs_identity.json"
    payload = _load_json(path)
    if payload.get("schema") != CAMPAIGN_INPUTS_SCHEMA:
        raise CampaignBuildError("campaign input identity schema differs")
    claimed = payload.get("identity_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if claimed != canonical_sha256(unsigned):
        raise CampaignBuildError("campaign input identity self digest differs")
    if payload.get("code") != _code_manifest(CODE_ROOT):
        raise CampaignBuildError("frozen code content differs from input identity")
    pinned = payload.get("pinned_files")
    if not isinstance(pinned, Mapping):
        raise CampaignBuildError("campaign pinned-file identity is malformed")
    for relative, expected in pinned.items():
        file_path = RUN_ROOT / "inputs" / str(relative)
        if not isinstance(expected, Mapping):
            raise CampaignBuildError("campaign pinned-file row is malformed")
        if (
            file_path.stat().st_size != expected.get("size")
            or _sha256_file(file_path) != expected.get("sha256")
        ):
            raise CampaignBuildError(f"campaign pinned file differs: {file_path}")
    return payload


def host_preflight(*, host_id: str, commit: str) -> None:
    image_id, gpu_uuid, minimum_free = _host_config(host_id)
    _verify_inputs_identity()
    _verify_source(full_content=True)
    if shutil.disk_usage(RUN_ROOT).free < minimum_free:
        raise CampaignBuildError(
            f"{host_id} free disk is below pinned floor {minimum_free}"
        )
    _verify_image(image_id)
    _verify_gpu(gpu_uuid)
    internal = [
        *_docker_prefix(image_id, commit, gpu=True),
        "/campaign/code/tools/build_prismasnap_27b_campaign.py",
        "container-preflight",
        "--host",
        host_id,
        "--producer-identity",
        str(RUN_ROOT / "inputs" / "producer_identity.json"),
    ]
    subprocess.run(internal, check=True)


def container_preflight(*, host_id: str, producer_identity: Path) -> None:
    expected = _load_json(producer_identity)
    from prismaquant.prismasnap_checkpoint import _producer_identity
    import torch

    observed = _producer_identity()
    if expected != observed:
        raise CampaignBuildError("container PrismaSnap producer identity differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise CampaignBuildError("container must see exactly one CUDA GPU")
    _image, expected_uuid, _floor = _host_config(host_id)
    properties = torch.cuda.get_device_properties(0)
    if properties.name != GPU_NAME or f"{properties.major}.{properties.minor}" != GPU_COMPUTE_CAPABILITY:
        raise CampaignBuildError("container CUDA device identity differs")
    # torch does not expose a stable UUID on every build; the host-side
    # nvidia-smi preflight binds the exact UUID before this container starts.
    del expected_uuid


def _sealed_stage(
    *,
    stage_id: str,
    host_id: str,
    dependencies: Sequence[str],
    child_argv: Sequence[str],
    timeout_seconds: int,
    max_attempts: int,
) -> dict[str, object]:
    token = f"{CAMPAIGN_ID}:{stage_id}"
    receipt = RUN_ROOT / "campaign" / "receipts" / f"{stage_id}.sealed.json"
    child = list(child_argv)
    wrapper = [
        "/usr/bin/python3",
        "-B",
        str(CODE_ROOT / "prismaquant" / "cluster_campaign.py"),
        "sealed-stage",
        "--receipt",
        str(receipt),
        "--token",
        token,
        "--",
        *child,
    ]
    return {
        "id": stage_id,
        "host_id": host_id,
        "dependencies": list(dependencies),
        "argv": wrapper,
        "cwd": str(CODE_ROOT),
        "env": {
            "HOME": "/home/rob",
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        },
        "receipts": [
            {
                "path": str(receipt),
                "sha256": sealed_stage_receipt_sha256(token, child),
            }
        ],
        "max_attempts": max_attempts,
        "timeout_seconds": timeout_seconds,
    }


def _transfer_child(
    *, direction: str, source: Path, destination: Path, stage_id: str
) -> list[str]:
    return [
        "/usr/bin/python3",
        "-B",
        str(CODE_ROOT / "tools" / Path(__file__).name),
        "transfer-tree",
        "--direction",
        direction,
        "--source",
        str(source),
        "--destination",
        str(destination),
        "--receipt",
        str(RUN_ROOT / "campaign" / "transfers" / f"{stage_id}.json"),
    ]


def build_campaign_body(commit: str) -> dict[str, object]:
    """Return the exact unsealed campaign body for a frozen producer commit."""
    if not _COMMIT_RE.fullmatch(commit):
        raise CampaignBuildError("producer commit must be a full lowercase 40-hex ID")
    tool = str(CODE_ROOT / "tools" / Path(__file__).name)
    source_identity = str(RUN_ROOT / "inputs" / "source_model_identity.json")
    original_probe = str(RUN_ROOT / "inputs" / "original-probe.pkl")
    normalized_probe = str(RUN_ROOT / "probe" / "normalized-probe.pkl")
    headers = str(RUN_ROOT / "probe" / "tensor-metadata-manifest.json")
    stages: list[dict[str, object]] = []

    for host_id in ("sparky", "sparklina"):
        stages.append(
            _sealed_stage(
                stage_id=f"preflight-{host_id}",
                host_id=host_id,
                dependencies=[],
                child_argv=[
                    "/usr/bin/python3",
                    "-B",
                    tool,
                    "host-preflight",
                    "--host",
                    host_id,
                    "--producer-git-commit",
                    commit,
                ],
                timeout_seconds=600,
                max_attempts=1,
            )
        )

    stages.append(
        _sealed_stage(
            stage_id="bind-probe",
            host_id="sparky",
            dependencies=["preflight-sparky", "preflight-sparklina"],
            child_argv=_prismasnap_argv(
                LOCAL_IMAGE_ID,
                commit,
                [
                    "bind-legacy-text-probe",
                    "--source",
                    str(HF_ALIAS),
                    "--source-identity",
                    source_identity,
                    "--probe",
                    original_probe,
                    "--output",
                    normalized_probe,
                    "--resume",
                ],
                gpu=False,
            ),
            timeout_seconds=7200,
            max_attempts=2,
        )
    )
    stages.append(
        _sealed_stage(
            stage_id="scan-tensor-metadata",
            host_id="sparky",
            dependencies=["bind-probe"],
            child_argv=_prismasnap_argv(
                LOCAL_IMAGE_ID,
                commit,
                [
                    "scan-tensor-metadata",
                    "--source",
                    str(HF_ALIAS),
                    "--source-identity",
                    source_identity,
                    "--output",
                    headers,
                    "--resume",
                ],
                gpu=False,
            ),
            timeout_seconds=7200,
            max_attempts=2,
        )
    )
    stages.append(
        _sealed_stage(
            stage_id="push-planning-inputs",
            host_id="sparky",
            dependencies=["scan-tensor-metadata"],
            child_argv=_transfer_child(
                direction="push",
                source=RUN_ROOT / "probe",
                destination=RUN_ROOT / "probe",
                stage_id="push-planning-inputs",
            ),
            timeout_seconds=1800,
            max_attempts=3,
        )
    )

    for host_id, image_id, layers in (
        ("sparky", LOCAL_IMAGE_ID, "0-31"),
        ("sparklina", REMOTE_IMAGE_ID, "32-63"),
    ):
        stages.append(
            _sealed_stage(
                stage_id=f"plan-{host_id}",
                host_id=host_id,
                dependencies=["push-planning-inputs"],
                child_argv=_prismasnap_argv(
                    image_id,
                    commit,
                    [
                        "plan-dense",
                        "--source",
                        str(HF_ALIAS),
                        "--probe",
                        normalized_probe,
                        "--source-identity",
                        source_identity,
                        "--tensor-metadata-manifest",
                        headers,
                        "--output",
                        str(RUN_ROOT / "plans" / host_id),
                        "--layers",
                        layers,
                        "--device",
                        "cuda",
                        "--alphas",
                        "0.0",
                        "0.125",
                        "0.25",
                        "0.375",
                        "0.5",
                        "--max-rounds",
                        "4",
                        "--polish-top",
                        "8",
                        "--polish-pool",
                        "16",
                        "--nvfp4-scale-rule",
                        "static_6",
                        "--resume",
                    ],
                    gpu=True,
                ),
                timeout_seconds=43_200,
                max_attempts=2,
            )
        )

    stages.append(
        _sealed_stage(
            stage_id="fetch-plan-sparklina",
            host_id="sparky",
            dependencies=["plan-sparky", "plan-sparklina"],
            child_argv=_transfer_child(
                direction="fetch",
                source=RUN_ROOT / "plans" / "sparklina",
                destination=RUN_ROOT / "plans" / "sparklina",
                stage_id="fetch-plan-sparklina",
            ),
            timeout_seconds=3600,
            max_attempts=3,
        )
    )
    stages.append(
        _sealed_stage(
            stage_id="merge-plans",
            host_id="sparky",
            dependencies=["fetch-plan-sparklina"],
            child_argv=_prismasnap_argv(
                LOCAL_IMAGE_ID,
                commit,
                [
                    "merge-plans",
                    "--input",
                    str(RUN_ROOT / "plans" / "sparky"),
                    "--input",
                    str(RUN_ROOT / "plans" / "sparklina"),
                    "--output",
                    str(RUN_ROOT / "plans" / "merged"),
                    "--resume",
                ],
                gpu=False,
            ),
            timeout_seconds=1800,
            max_attempts=2,
        )
    )
    stages.append(
        _sealed_stage(
            stage_id="push-merged-plan",
            host_id="sparky",
            dependencies=["merge-plans"],
            child_argv=_transfer_child(
                direction="push",
                source=RUN_ROOT / "plans" / "merged",
                destination=RUN_ROOT / "plans" / "merged",
                stage_id="push-merged-plan",
            ),
            timeout_seconds=1800,
            max_attempts=3,
        )
    )
    for host_id, image_id in (
        ("sparky", LOCAL_IMAGE_ID),
        ("sparklina", REMOTE_IMAGE_ID),
    ):
        stages.append(
            _sealed_stage(
                stage_id=f"materialize-part-{host_id}",
                host_id=host_id,
                dependencies=["push-merged-plan"],
                child_argv=_prismasnap_argv(
                    image_id,
                    commit,
                    [
                        "materialize-part",
                        "--source",
                        str(HF_ALIAS),
                        "--plan",
                        str(RUN_ROOT / "plans" / "merged"),
                        "--output",
                        str(RUN_ROOT / "parts" / host_id),
                        "--shards-file",
                        str(RUN_ROOT / "inputs" / f"shards-{host_id}.txt"),
                        "--device",
                        "cuda",
                        "--resume",
                    ],
                    gpu=True,
                ),
                timeout_seconds=14_400,
                max_attempts=2,
            )
        )
    stages.append(
        _sealed_stage(
            stage_id="fetch-part-sparklina",
            host_id="sparky",
            dependencies=[
                "materialize-part-sparky",
                "materialize-part-sparklina",
            ],
            child_argv=_transfer_child(
                direction="fetch",
                source=RUN_ROOT / "parts" / "sparklina",
                destination=RUN_ROOT / "parts" / "sparklina",
                stage_id="fetch-part-sparklina",
            ),
            timeout_seconds=7200,
            max_attempts=3,
        )
    )
    stages.append(
        _sealed_stage(
            stage_id="merge-checkpoint-parts",
            host_id="sparky",
            dependencies=["fetch-part-sparklina"],
            child_argv=_prismasnap_argv(
                LOCAL_IMAGE_ID,
                commit,
                [
                    "merge-checkpoint-parts",
                    "--source",
                    str(HF_ALIAS),
                    "--plan",
                    str(RUN_ROOT / "plans" / "merged"),
                    "--part",
                    str(RUN_ROOT / "parts" / "sparky"),
                    "--part",
                    str(RUN_ROOT / "parts" / "sparklina"),
                    "--output",
                    str(RUN_ROOT / "snapped-source"),
                    "--resume",
                    "--require-hardlinks",
                ],
                gpu=False,
            ),
            timeout_seconds=14_400,
            max_attempts=2,
        )
    )
    return {
        "schema": CAMPAIGN_MANIFEST_SCHEMA_V2,
        "campaign_id": CAMPAIGN_ID,
        "coordinator": "sparky",
        "max_parallel": 2,
        "hosts": [
            {
                "id": "sparky",
                "transport": {"kind": "local"},
                "work_root": str(RUN_ROOT / "campaign" / "workers" / "sparky"),
            },
            {
                "id": "sparklina",
                "transport": {
                    "kind": "ssh",
                    "host": REMOTE_ADDRESS,
                    "port": 22,
                    "user": REMOTE_USER,
                    "known_hosts": str(KNOWN_HOSTS),
                    "ssh_executable": str(SSH_SHIM),
                    "remote_helper_argv": [
                        "/usr/bin/python3",
                        "-B",
                        str(CODE_ROOT / "prismaquant" / "cluster_campaign.py"),
                        "_exec-request",
                    ],
                    "connect_timeout_seconds": 20,
                },
                "work_root": str(RUN_ROOT / "campaign" / "workers" / "sparklina"),
            },
        ],
        "stages": stages,
    }


def _git_commit_and_clean(repository: Path, expected_commit: str) -> None:
    completed = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.stdout.strip().lower() != expected_commit:
        raise CampaignBuildError("repository HEAD differs from declared producer commit")
    status_result = subprocess.run(
        ["/usr/bin/git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if status_result.stdout:
        raise CampaignBuildError("repository is not clean; refusing to freeze producer")


def _prepare_code(repository: Path) -> dict[str, object]:
    repository = repository.resolve(strict=True)
    if repository == CODE_ROOT.resolve(strict=False):
        return _code_manifest(repository)
    expected = _code_manifest(repository)
    if CODE_ROOT.exists():
        observed = _code_manifest(CODE_ROOT)
        if observed != expected:
            raise CampaignBuildError(
                "existing frozen code differs; refusing to overwrite immutable snapshot"
            )
        return observed
    CODE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = CODE_ROOT.with_name(f".{CODE_ROOT.name}.prepare")
    if os.path.lexists(temporary):
        raise CampaignBuildError(f"stale code snapshot temporary exists: {temporary}")
    temporary.mkdir()
    try:
        for name in CODE_SUBTREES:
            shutil.copytree(
                repository / name,
                temporary / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        if _code_manifest(temporary) != expected:
            raise CampaignBuildError("copied code snapshot differs from source")
        os.replace(temporary, CODE_ROOT)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return expected


def _emit_producer_identity(commit: str) -> dict[str, Any]:
    command = [
        *_docker_prefix(LOCAL_IMAGE_ID, commit, gpu=False),
        "/campaign/code/tools/build_prismasnap_27b_campaign.py",
        "emit-producer-identity",
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise CampaignBuildError("container producer identity is not an object")
    if (
        value.get("git_commit") != commit
        or value.get("container_rootfs_sha256") != ROOTFS_LAYERS_SHA256
        or value.get("container_attested") is not True
    ):
        raise CampaignBuildError("container producer identity differs from pins")
    return value


def _pinned_file_rows() -> dict[str, dict[str, object]]:
    names = (
        "original-probe.pkl",
        "source_model_identity.json",
        "shards-sparky.txt",
        "shards-sparklina.txt",
        "producer_identity.json",
        "ssh-sparklina",
    )
    return {
        name: {
            "size": (RUN_ROOT / "inputs" / name).stat().st_size,
            "sha256": _sha256_file(RUN_ROOT / "inputs" / name),
        }
        for name in names
    }


def _sync_bootstrap_inputs() -> None:
    # The explicit shim ignores ambient SSH aliases/config.  Source and target
    # paths are identical so frozen probe receipts remain portable.
    remote = f"{REMOTE_USER}@{REMOTE_ADDRESS}"
    common = [
        "/usr/bin/rsync",
        "--archive",
        "--checksum",
        "--partial",
        "--protect-args",
        "--mkpath",
        "--rsync-path=/usr/bin/rsync",
        f"--rsh={SSH_SHIM}",
    ]
    for name in CODE_SUBTREES:
        subprocess.run(
            [
                *common,
                str(CODE_ROOT / name) + "/",
                f"{remote}:{CODE_ROOT / name}/",
            ],
            check=True,
        )
    files = [
        RUN_ROOT / "inputs" / name
        for name in (
            "original-probe.pkl",
            "source_model_identity.json",
            "shards-sparky.txt",
            "shards-sparklina.txt",
            "producer_identity.json",
            "ssh-sparklina",
            "campaign_inputs_identity.json",
        )
    ]
    subprocess.run(
        [*common, *(str(path) for path in files), f"{remote}:{RUN_ROOT / 'inputs'}/"],
        check=True,
    )
    remote_command = [
        *_ssh_base(),
        "/usr/bin/python3",
        "-B",
        str(CODE_ROOT / "tools" / Path(__file__).name),
        "verify-inputs",
    ]
    subprocess.run(remote_command, check=True)


def build(*, commit: str, repository: Path) -> dict[str, object]:
    if not _COMMIT_RE.fullmatch(commit):
        raise CampaignBuildError("producer commit must be a full lowercase 40-hex ID")
    _git_commit_and_clean(repository, commit)
    code = _prepare_code(repository)
    for directory in (
        RUN_ROOT / "probe",
        RUN_ROOT / "plans",
        RUN_ROOT / "parts",
        RUN_ROOT / "campaign" / "receipts",
        RUN_ROOT / "campaign" / "transfers",
    ):
        if os.path.lexists(directory) and (directory.is_symlink() or not directory.is_dir()):
            raise CampaignBuildError(f"campaign directory is unsafe: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    _copy_or_verify(
        ORIGINAL_PROBE_SOURCE,
        RUN_ROOT / "inputs" / "original-probe.pkl",
        ORIGINAL_PROBE_SHA256,
    )
    _copy_or_verify(
        SOURCE_IDENTITY_SOURCE,
        RUN_ROOT / "inputs" / "source_model_identity.json",
        SOURCE_IDENTITY_SHA256,
    )
    _atomic_write_or_verify(
        RUN_ROOT / "inputs" / "shards-sparky.txt",
        ("\n".join(SPARKY_SHARDS) + "\n").encode("utf-8"),
    )
    _atomic_write_or_verify(
        RUN_ROOT / "inputs" / "shards-sparklina.txt",
        ("\n".join(SPARKLINA_SHARDS) + "\n").encode("utf-8"),
    )
    _atomic_write_or_verify(SSH_SHIM, _ssh_shim_bytes(), mode=0o500)
    _verify_image(LOCAL_IMAGE_ID)
    producer = _emit_producer_identity(commit)
    _atomic_write_or_verify(
        RUN_ROOT / "inputs" / "producer_identity.json",
        _pretty_bytes(producer),
    )
    unsigned_inputs = {
        "schema": CAMPAIGN_INPUTS_SCHEMA,
        "campaign_builder_schema": CAMPAIGN_BUILDER_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "producer_git_commit": commit,
        "rootfs_layers_sha256": ROOTFS_LAYERS_SHA256,
        "host_image_ids": {
            "sparky": LOCAL_IMAGE_ID,
            "sparklina": REMOTE_IMAGE_ID,
        },
        "source_root": str(SOURCE_ROOT),
        "source_portable_sha256": SOURCE_PORTABLE_SHA256,
        "code": code,
        "pinned_files": _pinned_file_rows(),
    }
    input_identity = {
        **unsigned_inputs,
        "identity_sha256": canonical_sha256(unsigned_inputs),
    }
    _atomic_write_or_verify(
        RUN_ROOT / "inputs" / "campaign_inputs_identity.json",
        _pretty_bytes(input_identity),
    )
    _verify_inputs_identity()
    _verify_source(full_content=False)
    _sync_bootstrap_inputs()
    body = build_campaign_body(commit)
    sealed = seal_campaign_manifest_v2(body)
    _atomic_write_or_verify(
        RUN_ROOT / "campaign" / "manifest.body.json", _pretty_bytes(body)
    )
    _atomic_write_or_verify(
        RUN_ROOT / "campaign" / "manifest.json", _pretty_bytes(sealed)
    )
    return {
        "schema": CAMPAIGN_BUILDER_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "producer_git_commit": commit,
        "campaign_identity_sha256": sealed["identity_sha256"],
        "manifest": str(RUN_ROOT / "campaign" / "manifest.json"),
        "state": str(RUN_ROOT / "campaign" / "state.json"),
        "launched": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--producer-git-commit", required=True)
    build_parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    preflight = sub.add_parser("host-preflight")
    preflight.add_argument("--host", choices=("sparky", "sparklina"), required=True)
    preflight.add_argument("--producer-git-commit", required=True)
    container = sub.add_parser("container-preflight")
    container.add_argument("--host", choices=("sparky", "sparklina"), required=True)
    container.add_argument("--producer-identity", type=Path, required=True)
    transfer = sub.add_parser("transfer-tree")
    transfer.add_argument("--direction", choices=("push", "fetch"), required=True)
    transfer.add_argument("--source", type=Path, required=True)
    transfer.add_argument("--destination", type=Path, required=True)
    transfer.add_argument("--receipt", type=Path, required=True)
    tree = sub.add_parser("hash-tree")
    tree.add_argument("--path", type=Path, required=True)
    sub.add_parser("verify-inputs")
    sub.add_parser("emit-producer-identity")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            payload = build(
                commit=args.producer_git_commit.lower(), repository=args.repository_root
            )
        elif args.command == "host-preflight":
            host_preflight(
                host_id=args.host, commit=args.producer_git_commit.lower()
            )
            payload = {"host": args.host, "preflight": "passed"}
        elif args.command == "container-preflight":
            container_preflight(
                host_id=args.host, producer_identity=args.producer_identity
            )
            payload = {"host": args.host, "container_preflight": "passed"}
        elif args.command == "transfer-tree":
            payload = transfer_tree(
                direction=args.direction,
                source=args.source,
                destination=args.destination,
                receipt=args.receipt,
            )
        elif args.command == "hash-tree":
            payload = _tree_manifest(args.path)
        elif args.command == "verify-inputs":
            payload = _verify_inputs_identity()
            _verify_source(full_content=False)
        elif args.command == "emit-producer-identity":
            from prismaquant.prismasnap_checkpoint import _producer_identity

            payload = _producer_identity()
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (CampaignBuildError, OSError, subprocess.SubprocessError) as exc:
        print(f"prismasnap-27b-campaign: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
