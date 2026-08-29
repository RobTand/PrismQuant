"""Pure, identity-bound state machine for the two-host RTX 4090 campaign.

This module deliberately does not perform I/O, launch subprocesses, or know
how an SSH transport is implemented.  It validates one closed campaign
manifest and advances a durable state through a fixed sequence of logical
stages.  A stage scoped to ``all_hosts`` is a barrier: both host assignments
must complete before a dependent stage becomes ready.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Literal

from prismaquant.cost_stage_checkpoint import canonical_json, canonical_json_sha256


LEGACY_CAMPAIGN_MANIFEST_SCHEMA = "prismaquant.cluster_campaign.manifest.v2"
CAMPAIGN_MANIFEST_SCHEMA = "prismaquant.cluster_campaign.manifest.v3"
CAMPAIGN_STATE_SCHEMA = "prismaquant.cluster_campaign.state.v1"
STAGE_RECEIPT_SCHEMA = "prismaquant.cluster_campaign.stage_receipt.v1"
GRIDBOOK_RUNTIME_CONTRACT_INPUT_SCHEMA = (
    "prismaquant.cluster_campaign.gridbook_runtime_contract_input.v1"
)

# These identify the artifact's target lane, not the hardware that executes a
# validation campaign. In particular, the reviewed RTX 4090 producer is
# exercised on GB10 ([12, 1]) before physical Ada qualification.
RTX4090_GPU_NAME = "NVIDIA GeForce RTX 4090"
RTX4090_COMPUTE_CAPABILITY = (8, 9)

CANONICAL_CONTAINER_PATHS = MappingProxyType(
    {
        "model_root": "/model",
        "dataset_path": "/dataset",
        "snapshot_root": "/pq",
        "run_root": "/run",
        "worker_state_root": "/worker-state",
    }
)


class ClusterCampaignContractError(ValueError):
    """A campaign manifest, state, or requested transition is invalid."""


StageScope = Literal["coordinator", "all_hosts"]


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One logical stage in the fixed campaign DAG."""

    stage: str
    scope: StageScope
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StageAssignment:
    """One host's deterministic unit of work in a logical stage."""

    work_id: str
    stage: str
    host_id: str
    assignment_index: int


def _chain_stage(
    stage: str,
    scope: StageScope,
    previous: str | None,
) -> StageSpec:
    return StageSpec(
        stage=stage,
        scope=scope,
        dependencies=() if previous is None else (previous,),
    )


# The order is part of the public manifest contract. Parallelism exists only inside
# all-host stages; the explicit dependency at every boundary keeps recovery
# and orchestration deterministic.
STAGE_DAG: tuple[StageSpec, ...] = (
    _chain_stage("host_preflight", "all_hosts", None),
    _chain_stage("prepare_calibration", "coordinator", "host_preflight"),
    _chain_stage(
        "coordinator_source_identity",
        "coordinator",
        "prepare_calibration",
    ),
    _chain_stage(
        "prepare_run_contract",
        "coordinator",
        "coordinator_source_identity",
    ),
    _chain_stage("build_sample_cover", "coordinator", "prepare_run_contract"),
    _chain_stage(
        "worker_source_identity",
        "all_hosts",
        "build_sample_cover",
    ),
    _chain_stage("sample_ce", "all_hosts", "worker_source_identity"),
    _chain_stage("merge_importance", "coordinator", "sample_ce"),
    _chain_stage("sample_fisher", "all_hosts", "merge_importance"),
    _chain_stage("merge_sample_probe", "coordinator", "sample_fisher"),
    _chain_stage("derive_col_weights", "coordinator", "merge_sample_probe"),
    _chain_stage("attest_execution", "coordinator", "derive_col_weights"),
    _chain_stage("prepare_burn", "coordinator", "attest_execution"),
    _chain_stage("measure_burn", "all_hosts", "prepare_burn"),
    _chain_stage("merge_burn", "coordinator", "measure_burn"),
    _chain_stage("allocate", "coordinator", "merge_burn"),
    _chain_stage(
        "export_validation_artifact", "coordinator", "allocate",
    ),
    _chain_stage(
        "qualify_validation_artifact",
        "coordinator",
        "export_validation_artifact",
    ),
)

_STAGE_BY_NAME = {item.stage: item for item in STAGE_DAG}
if len(_STAGE_BY_NAME) != len(STAGE_DAG):  # pragma: no cover - static guard
    raise RuntimeError("cluster campaign DAG contains duplicate stage names")

_MANIFEST_BODY_KEYS = frozenset(
    {
        "schema", "campaign_id", "coordinator", "artifact_target",
        "producer", "inputs", "policy", "hosts",
    }
)
_MANIFEST_KEYS = _MANIFEST_BODY_KEYS | {"identity_sha256"}
_PRODUCER_KEYS = frozenset(
    {"commit", "tree", "snapshot_sha256", "image_digest"}
)
_ARTIFACT_TARGET_KEYS = frozenset({
    "gpu_name",
    "compute_capability",
    "artifact_max_bytes",
    "disposition",
    "source_dtype",
    "physical_formats",
    "terminal_format",
    "allocation_objective",
})
_INPUT_KEYS = frozenset({
    "model_content_sha256",
    "dataset_sha256",
    "sample_parallel",
    "gridbook_runtime_contract",
})
_SAMPLE_KEYS = frozenset(
    {"nsamples", "seqlen", "calib_seed", "activation_rows_limit"}
)
_GRIDBOOK_RUNTIME_CONTRACT_INPUT_KEYS = frozenset({
    "schema", "runtime_contract_sha256", "payload",
})
_POLICY_KEYS = frozenset({"retry", "telemetry", "resources", "outputs"})
_RETRY_KEYS = frozenset({"max_attempts"})
_LEGACY_TELEMETRY_KEYS = frozenset(
    {"interval_milliseconds", "require_positive_gpu_utilization"}
)
_TELEMETRY_KEYS = frozenset(
    {
        "interval_milliseconds",
        "maximum_observation_gap_milliseconds",
        "minimum_successful_sample_percent",
        "require_positive_gpu_utilization",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "coordinator_min_free_bytes", "worker_min_free_bytes",
        "min_mem_available_bytes",
    }
)
_OUTPUT_KEYS = frozenset({"owner", "transfer_mode"})
_FIXED_SAMPLE_PARALLEL = MappingProxyType(
    {
        "nsamples": 32,
        "seqlen": 1024,
        "calib_seed": 42,
        "activation_rows_limit": 1024,
    }
)
FIXED_CAMPAIGN_POLICY = MappingProxyType(
    {
        "retry": {"max_attempts": 2},
        "telemetry": {
            "interval_milliseconds": 1000,
            "maximum_observation_gap_milliseconds": 30_000,
            "minimum_successful_sample_percent": 50,
            "require_positive_gpu_utilization": True,
        },
        "resources": {
            "coordinator_min_free_bytes": 100 * 1024**3,
            "worker_min_free_bytes": 40 * 1024**3,
            "min_mem_available_bytes": 64 * 1024**3,
        },
        "outputs": {
            "owner": "coordinator",
            "transfer_mode": "sha256_no_clobber",
        },
    }
)
LEGACY_FIXED_CAMPAIGN_POLICY = MappingProxyType(
    {
        "retry": {"max_attempts": 2},
        "telemetry": {
            "interval_milliseconds": 1000,
            "require_positive_gpu_utilization": True,
        },
        "resources": {
            "coordinator_min_free_bytes": 100 * 1024**3,
            "worker_min_free_bytes": 40 * 1024**3,
            "min_mem_available_bytes": 64 * 1024**3,
        },
        "outputs": {
            "owner": "coordinator",
            "transfer_mode": "sha256_no_clobber",
        },
    }
)
_HOST_KEYS = frozenset({"id", "transport", "roots", "expected"})
_ROOT_KEYS = frozenset(CANONICAL_CONTAINER_PATHS)
_EXPECTED_KEYS = frozenset(
    {
        "hostname",
        "gpu",
        "image_digest",
        "producer_commit",
        "uid",
        "gid",
    }
)
_GPU_KEYS = frozenset(
    {"name", "uuid", "compute_capability", "device_count"}
)
_STATE_KEYS = frozenset(
    {"schema", "campaign_identity_sha256", "completions", "identity_sha256"}
)
_COMPLETION_KEYS = frozenset(
    {
        "schema",
        "work_id",
        "stage",
        "host_id",
        "assignment_index",
        "campaign_identity_sha256",
        "host_identity_sha256",
        "producer_identity_sha256",
        "receipt_sha256",
    }
)

_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_HOST_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,252}\Z")
_USER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")
_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9._-]+\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
_IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")

FIXED_ARTIFACT_TARGET = MappingProxyType({
    "gpu_name": RTX4090_GPU_NAME,
    "compute_capability": RTX4090_COMPUTE_CAPABILITY,
    "artifact_max_bytes": 18_000_000_000,
    "disposition": "validation_only",
    "source_dtype": "bf16",
    "physical_formats": (
        "FP8_CB_K40", "FP8_CB_K44", "FP8_CB_K48", "FP8_E4M3",
    ),
    "terminal_format": "BF16",
    "allocation_objective": "context_first",
})

# Manifest v2 is sealed historical evidence.  Its K4/K16/K48 numerical target
# must remain byte-exact and verifiable even though those low rungs no longer
# belong to the fresh producer registry.  This constant is read-only
# compatibility authority; new manifests can never select it.
LEGACY_FIXED_ARTIFACT_TARGET = MappingProxyType({
    "gpu_name": RTX4090_GPU_NAME,
    "compute_capability": RTX4090_COMPUTE_CAPABILITY,
    "artifact_max_bytes": 18_000_000_000,
    "disposition": "validation_only",
    "source_dtype": "bf16",
    "physical_formats": (
        "FP8_CB_K4", "FP8_CB_K16", "FP8_CB_K48", "FP8_E4M3",
    ),
    "terminal_format": "BF16",
    "allocation_objective": "context_first",
})


def _fail(message: str) -> None:
    raise ClusterCampaignContractError(message)


def _exact_mapping(
    value: object,
    *,
    keys: frozenset[str],
    where: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be an object")
    if any(type(key) is not str for key in value):
        _fail(f"{where} keys must be strings")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        _fail(f"{where} fields differ: missing={missing}, extra={extra}")
    return value


def _string(
    value: object,
    *,
    where: str,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if type(value) is not str or not value:
        _fail(f"{where} must be a non-empty string")
    if value != value.strip() or any(ord(char) < 32 for char in value):
        _fail(f"{where} contains whitespace padding or control characters")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(f"{where} has an invalid value")
    return value


def _integer(
    value: object,
    *,
    where: str,
    minimum: int,
    maximum: int = 2**63 - 1,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{where} must be an integer in [{minimum}, {maximum}]")
    return value


def _boolean(value: object, *, where: str) -> bool:
    if type(value) is not bool:
        _fail(f"{where} must be a boolean")
    return value


def _sha256(value: object, *, where: str) -> str:
    return _string(value, where=where, pattern=_SHA256_RE)


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 of compact, sorted, strict canonical JSON."""

    try:
        return canonical_json_sha256(value, where="cluster campaign value")
    except (TypeError, ValueError) as exc:
        raise ClusterCampaignContractError(
            "cluster campaign value is not canonical JSON data"
        ) from exc


def gridbook_runtime_contract_sha256(value: object) -> str:
    """Return the digest Gridbook producer attestations use for a contract."""

    try:
        canonical = canonical_json(
            value, where="Gridbook runtime contract input",
        )
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ClusterCampaignContractError(
            "Gridbook runtime contract input is not canonical JSON data"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def bind_gridbook_runtime_contract(value: Mapping[str, object]) -> dict[str, object]:
    """Embed one immutable external Gridbook contract in a campaign manifest."""

    if not isinstance(value, Mapping):
        _fail("Gridbook runtime contract input must be an object")
    try:
        payload = canonical_json(
            dict(value), where="Gridbook runtime contract input",
        )
    except (TypeError, ValueError) as exc:
        raise ClusterCampaignContractError(
            "Gridbook runtime contract input is not canonical JSON data"
        ) from exc
    assert isinstance(payload, dict)
    return {
        "schema": GRIDBOOK_RUNTIME_CONTRACT_INPUT_SCHEMA,
        "runtime_contract_sha256": gridbook_runtime_contract_sha256(payload),
        "payload": payload,
    }


def _safe_absolute_path(value: object, *, where: str) -> str:
    raw = _string(value, where=where)
    if not raw.startswith("/") or raw == "/":
        _fail(f"{where} must be a non-root absolute POSIX path")
    components = raw.split("/")[1:]
    if (
        not components
        or any(
            not component
            or component in {".", ".."}
            or _PATH_COMPONENT_RE.fullmatch(component) is None
            for component in components
        )
        or str(PurePosixPath(raw)) != raw
    ):
        _fail(f"{where} must be normalized and traversal-free")
    return raw


def _is_ancestor(left: str, right: str) -> bool:
    left_parts = PurePosixPath(left).parts
    right_parts = PurePosixPath(right).parts
    return len(left_parts) <= len(right_parts) and (
        right_parts[: len(left_parts)] == left_parts
    )


def _normalize_producer(value: object) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_PRODUCER_KEYS, where="producer")
    return {
        "commit": _string(
            raw["commit"], where="producer.commit", pattern=_GIT_SHA_RE
        ),
        "tree": _string(raw["tree"], where="producer.tree", pattern=_GIT_SHA_RE),
        "snapshot_sha256": _sha256(
            raw["snapshot_sha256"], where="producer.snapshot_sha256"
        ),
        "image_digest": _string(
            raw["image_digest"],
            where="producer.image_digest",
            pattern=_IMAGE_DIGEST_RE,
        ),
    }


def _normalize_artifact_target(
    value: object, *, manifest_schema: str,
) -> dict[str, object]:
    raw = _exact_mapping(
        value, keys=_ARTIFACT_TARGET_KEYS, where="artifact_target",
    )
    capability = raw["compute_capability"]
    formats = raw["physical_formats"]
    if type(capability) is not list or len(capability) != 2:
        _fail("artifact_target.compute_capability must be a two-integer list")
    if type(formats) is not list:
        _fail("artifact_target.physical_formats must be a list")
    normalized = {
        "gpu_name": _string(raw["gpu_name"], where="artifact_target.gpu_name"),
        "compute_capability": [
            _integer(
                capability[index],
                where=f"artifact_target.compute_capability[{index}]",
                minimum=0,
                maximum=99,
            )
            for index in range(2)
        ],
        "artifact_max_bytes": _integer(
            raw["artifact_max_bytes"],
            where="artifact_target.artifact_max_bytes",
            minimum=1,
        ),
        "disposition": _string(
            raw["disposition"], where="artifact_target.disposition",
        ),
        "source_dtype": _string(
            raw["source_dtype"], where="artifact_target.source_dtype",
        ),
        "physical_formats": [
            _string(item, where=f"artifact_target.physical_formats[{index}]")
            for index, item in enumerate(formats)
        ],
        "terminal_format": _string(
            raw["terminal_format"], where="artifact_target.terminal_format",
        ),
        "allocation_objective": _string(
            raw["allocation_objective"],
            where="artifact_target.allocation_objective",
        ),
    }
    expected_target = (
        LEGACY_FIXED_ARTIFACT_TARGET
        if manifest_schema == LEGACY_CAMPAIGN_MANIFEST_SCHEMA
        else FIXED_ARTIFACT_TARGET
    )
    fixed = {
        **dict(expected_target),
        "compute_capability": list(
            expected_target["compute_capability"]
        ),
        "physical_formats": list(expected_target["physical_formats"]),
    }
    if normalized != fixed:
        _fail(
            "artifact_target must equal the fixed RTX4090 validation target "
            f"{fixed}"
        )
    return normalized


def _normalize_inputs(value: object) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_INPUT_KEYS, where="inputs")
    sample = _exact_mapping(
        raw["sample_parallel"],
        keys=_SAMPLE_KEYS,
        where="inputs.sample_parallel",
    )
    normalized_sample = {
        "nsamples": _integer(
            sample["nsamples"],
            where="inputs.sample_parallel.nsamples",
            minimum=2,
        ),
        "seqlen": _integer(
            sample["seqlen"],
            where="inputs.sample_parallel.seqlen",
            minimum=2,
        ),
        "calib_seed": _integer(
            sample["calib_seed"],
            where="inputs.sample_parallel.calib_seed",
            minimum=0,
        ),
        "activation_rows_limit": _integer(
            sample["activation_rows_limit"],
            where="inputs.sample_parallel.activation_rows_limit",
            minimum=1,
        ),
    }
    if normalized_sample != dict(_FIXED_SAMPLE_PARALLEL):
        _fail(
            "inputs.sample_parallel must equal the fixed RTX4090 burn "
            f"contract {dict(_FIXED_SAMPLE_PARALLEL)}"
        )
    contract_input = _exact_mapping(
        raw["gridbook_runtime_contract"],
        keys=_GRIDBOOK_RUNTIME_CONTRACT_INPUT_KEYS,
        where="inputs.gridbook_runtime_contract",
    )
    if contract_input["schema"] != GRIDBOOK_RUNTIME_CONTRACT_INPUT_SCHEMA:
        _fail("inputs.gridbook_runtime_contract.schema is unsupported")
    try:
        contract_payload = canonical_json(
            contract_input["payload"],
            where="inputs.gridbook_runtime_contract.payload",
        )
    except (TypeError, ValueError) as exc:
        raise ClusterCampaignContractError(
            "inputs.gridbook_runtime_contract.payload is not canonical JSON data"
        ) from exc
    if not isinstance(contract_payload, dict):
        _fail("inputs.gridbook_runtime_contract.payload must be an object")
    if contract_payload.get("schema") != "gridbook.runtime-contract.v11":
        _fail(
            "inputs.gridbook_runtime_contract.payload must be a Gridbook v11 "
            "runtime contract"
        )
    contract_sha256 = _sha256(
        contract_input["runtime_contract_sha256"],
        where="inputs.gridbook_runtime_contract.runtime_contract_sha256",
    )
    if contract_sha256 != gridbook_runtime_contract_sha256(contract_payload):
        _fail(
            "inputs.gridbook_runtime_contract.runtime_contract_sha256 differs "
            "from its canonical payload"
        )
    return {
        "model_content_sha256": _sha256(
            raw["model_content_sha256"], where="inputs.model_content_sha256"
        ),
        "dataset_sha256": _sha256(
            raw["dataset_sha256"], where="inputs.dataset_sha256"
        ),
        "sample_parallel": normalized_sample,
        "gridbook_runtime_contract": {
            "schema": GRIDBOOK_RUNTIME_CONTRACT_INPUT_SCHEMA,
            "runtime_contract_sha256": contract_sha256,
            "payload": contract_payload,
        },
    }


def _normalize_policy(
    value: object, *, manifest_schema: str,
) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_POLICY_KEYS, where="policy")
    retry = _exact_mapping(raw["retry"], keys=_RETRY_KEYS, where="policy.retry")
    telemetry_keys = (
        _LEGACY_TELEMETRY_KEYS
        if manifest_schema == LEGACY_CAMPAIGN_MANIFEST_SCHEMA
        else _TELEMETRY_KEYS
    )
    telemetry = _exact_mapping(
        raw["telemetry"], keys=telemetry_keys, where="policy.telemetry",
    )
    resources = _exact_mapping(
        raw["resources"], keys=_RESOURCE_KEYS, where="policy.resources",
    )
    outputs = _exact_mapping(
        raw["outputs"], keys=_OUTPUT_KEYS, where="policy.outputs",
    )
    normalized = {
        "retry": {
            "max_attempts": _integer(
                retry["max_attempts"], where="policy.retry.max_attempts",
                minimum=1, maximum=16,
            ),
        },
        "telemetry": {
            "interval_milliseconds": _integer(
                telemetry["interval_milliseconds"],
                where="policy.telemetry.interval_milliseconds",
                minimum=100, maximum=60_000,
            ),
            "require_positive_gpu_utilization": _boolean(
                telemetry["require_positive_gpu_utilization"],
                where="policy.telemetry.require_positive_gpu_utilization",
            ),
        },
        "resources": {
            key: _integer(
                resources[key], where=f"policy.resources.{key}", minimum=1,
            )
            for key in sorted(_RESOURCE_KEYS)
        },
        "outputs": {
            "owner": _string(outputs["owner"], where="policy.outputs.owner"),
            "transfer_mode": _string(
                outputs["transfer_mode"],
                where="policy.outputs.transfer_mode",
            ),
        },
    }
    if manifest_schema == CAMPAIGN_MANIFEST_SCHEMA:
        normalized["telemetry"].update({
            "maximum_observation_gap_milliseconds": _integer(
                telemetry["maximum_observation_gap_milliseconds"],
                where=(
                    "policy.telemetry.maximum_observation_gap_milliseconds"
                ),
                minimum=100, maximum=3_600_000,
            ),
            "minimum_successful_sample_percent": _integer(
                telemetry["minimum_successful_sample_percent"],
                where="policy.telemetry.minimum_successful_sample_percent",
                minimum=1, maximum=100,
            ),
        })
    expected_policy = (
        LEGACY_FIXED_CAMPAIGN_POLICY
        if manifest_schema == LEGACY_CAMPAIGN_MANIFEST_SCHEMA
        else FIXED_CAMPAIGN_POLICY
    )
    if normalized != dict(expected_policy):
        _fail(
            "policy must equal the fixed two-host campaign policy "
            f"{dict(expected_policy)}"
        )
    return normalized


def _normalize_transport(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be an object")
    kind = value.get("kind")
    if kind == "local":
        raw = _exact_mapping(value, keys=frozenset({"kind"}), where=where)
        return {"kind": _string(raw["kind"], where=f"{where}.kind")}
    if kind == "ssh":
        raw = _exact_mapping(
            value,
            keys=frozenset({"kind", "host", "port", "user"}),
            where=where,
        )
        return {
            "kind": _string(raw["kind"], where=f"{where}.kind"),
            "host": _string(
                raw["host"], where=f"{where}.host", pattern=_HOST_TOKEN_RE
            ),
            "port": _integer(
                raw["port"], where=f"{where}.port", minimum=1, maximum=65535
            ),
            "user": _string(
                raw["user"], where=f"{where}.user", pattern=_USER_RE
            ),
        }
    _fail(f"{where}.kind must be exactly 'local' or 'ssh'")


def _normalize_roots(value: object, *, where: str) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_ROOT_KEYS, where=where)
    roots = {
        key: _safe_absolute_path(raw[key], where=f"{where}.{key}")
        for key in CANONICAL_CONTAINER_PATHS
    }
    if len(set(roots.values())) != len(roots):
        _fail(f"{where} paths must be distinct")
    writable = ("run_root", "worker_state_root")
    for writable_name in writable:
        writable_path = roots[writable_name]
        for other_name, other_path in roots.items():
            if other_name == writable_name:
                continue
            if _is_ancestor(writable_path, other_path) or _is_ancestor(
                other_path, writable_path
            ):
                _fail(
                    f"{where}.{writable_name} overlaps {where}.{other_name}"
                )
    return roots


def _normalize_gpu(value: object, *, where: str) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_GPU_KEYS, where=where)
    capability = raw["compute_capability"]
    if type(capability) is not list or len(capability) != 2:
        _fail(f"{where}.compute_capability must be a two-integer list")
    normalized_capability = [
        _integer(
            capability[0],
            where=f"{where}.compute_capability[0]",
            minimum=0,
            maximum=99,
        ),
        _integer(
            capability[1],
            where=f"{where}.compute_capability[1]",
            minimum=0,
            maximum=99,
        ),
    ]
    # This is the exact execution device admitted by host preflight. It need
    # not equal the artifact target: GB10 is the validation host for the RTX
    # 4090 lane, while physical sm89 remains a later qualification gate.
    name = _string(raw["name"], where=f"{where}.name")
    device_count = _integer(
        raw["device_count"], where=f"{where}.device_count", minimum=1, maximum=1
    )
    return {
        "name": name,
        "uuid": _string(
            raw["uuid"], where=f"{where}.uuid", pattern=_HOST_TOKEN_RE
        ),
        "compute_capability": normalized_capability,
        "device_count": device_count,
    }


def _normalize_expected(
    value: object,
    *,
    where: str,
    producer: Mapping[str, object],
) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_EXPECTED_KEYS, where=where)
    image_digest = _string(
        raw["image_digest"],
        where=f"{where}.image_digest",
        pattern=_IMAGE_DIGEST_RE,
    )
    producer_commit = _string(
        raw["producer_commit"],
        where=f"{where}.producer_commit",
        pattern=_GIT_SHA_RE,
    )
    if image_digest != producer["image_digest"]:
        _fail(f"{where}.image_digest differs from producer.image_digest")
    if producer_commit != producer["commit"]:
        _fail(f"{where}.producer_commit differs from producer.commit")
    return {
        "hostname": _string(
            raw["hostname"],
            where=f"{where}.hostname",
            pattern=_HOST_TOKEN_RE,
        ),
        "gpu": _normalize_gpu(raw["gpu"], where=f"{where}.gpu"),
        "image_digest": image_digest,
        "producer_commit": producer_commit,
        "uid": _integer(
            raw["uid"], where=f"{where}.uid", minimum=1, maximum=2**31 - 1
        ),
        "gid": _integer(
            raw["gid"], where=f"{where}.gid", minimum=0, maximum=2**31 - 1
        ),
    }


def _normalize_host(
    value: object,
    *,
    index: int,
    producer: Mapping[str, object],
) -> dict[str, object]:
    where = f"hosts[{index}]"
    raw = _exact_mapping(value, keys=_HOST_KEYS, where=where)
    return {
        "id": _string(raw["id"], where=f"{where}.id", pattern=_ID_RE),
        "transport": _normalize_transport(
            raw["transport"], where=f"{where}.transport"
        ),
        "roots": _normalize_roots(raw["roots"], where=f"{where}.roots"),
        "expected": _normalize_expected(
            raw["expected"], where=f"{where}.expected", producer=producer
        ),
    }


def _normalize_manifest_body(value: object) -> dict[str, object]:
    raw = _exact_mapping(
        value, keys=_MANIFEST_BODY_KEYS, where="campaign manifest body"
    )
    schema = raw["schema"]
    if type(schema) is not str or schema not in {
        LEGACY_CAMPAIGN_MANIFEST_SCHEMA,
        CAMPAIGN_MANIFEST_SCHEMA,
    }:
        _fail("campaign manifest schema is unsupported")
    campaign_id = _string(
        raw["campaign_id"], where="campaign_id", pattern=_ID_RE
    )
    coordinator = _string(
        raw["coordinator"], where="coordinator", pattern=_ID_RE
    )
    artifact_target = _normalize_artifact_target(
        raw["artifact_target"], manifest_schema=str(schema),
    )
    producer = _normalize_producer(raw["producer"])
    inputs = _normalize_inputs(raw["inputs"])
    policy = _normalize_policy(raw["policy"], manifest_schema=str(schema))
    raw_hosts = raw["hosts"]
    if type(raw_hosts) is not list or len(raw_hosts) != 2:
        _fail("hosts must be a list containing exactly two hosts")
    hosts = [
        _normalize_host(host, index=index, producer=producer)
        for index, host in enumerate(raw_hosts)
    ]
    host_ids = [str(host["id"]) for host in hosts]
    if len(set(host_ids)) != len(host_ids):
        _fail("hosts contain duplicate ids")
    hostnames = [
        str(host["expected"]["hostname"])  # type: ignore[index]
        for host in hosts
    ]
    if len(set(hostnames)) != len(hostnames):
        _fail("hosts contain duplicate expected hostnames")
    gpu_uuids = [
        str(host["expected"]["gpu"]["uuid"])  # type: ignore[index]
        for host in hosts
    ]
    if len(set(gpu_uuids)) != len(gpu_uuids):
        _fail("hosts contain duplicate expected GPU UUIDs")
    kinds = [str(host["transport"]["kind"]) for host in hosts]  # type: ignore[index]
    if sorted(kinds) != ["local", "ssh"]:
        _fail("hosts must contain exactly one local and one ssh transport")
    if coordinator not in host_ids:
        _fail("coordinator does not reference a campaign host")
    hosts.sort(key=lambda item: str(item["id"]))
    return {
        "schema": schema,
        "campaign_id": campaign_id,
        "coordinator": coordinator,
        "artifact_target": artifact_target,
        "producer": producer,
        "inputs": inputs,
        "policy": policy,
        "hosts": hosts,
    }


def seal_campaign_manifest(body: Mapping[str, object]) -> dict[str, object]:
    """Validate a manifest body, canonicalize host order, and add its digest."""

    if (
        not isinstance(body, Mapping)
        or body.get("schema") != CAMPAIGN_MANIFEST_SCHEMA
    ):
        _fail("new campaign manifests must use the current schema")
    normalized = _normalize_manifest_body(body)
    return {**normalized, "identity_sha256": canonical_sha256(normalized)}


def validate_campaign_manifest(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and return the canonical representation of a sealed manifest."""

    raw = _exact_mapping(value, keys=_MANIFEST_KEYS, where="campaign manifest")
    body = _normalize_manifest_body(
        {key: raw[key] for key in _MANIFEST_BODY_KEYS}
    )
    identity = _sha256(raw["identity_sha256"], where="manifest.identity_sha256")
    expected = canonical_sha256(body)
    if identity != expected:
        _fail("manifest identity_sha256 differs from its canonical body")
    return {**body, "identity_sha256": identity}


def parse_campaign_manifest(text: str) -> dict[str, object]:
    """Strictly decode JSON (including duplicate rejection) and validate it."""

    if type(text) is not str:
        _fail("campaign manifest JSON must be text")

    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                _fail(f"campaign manifest contains duplicate JSON member {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> object:
        _fail(f"campaign manifest contains non-JSON constant {value}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except ClusterCampaignContractError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ClusterCampaignContractError(
            "campaign manifest is not valid strict JSON"
        ) from exc
    if not isinstance(decoded, Mapping):
        _fail("campaign manifest JSON root must be an object")
    return validate_campaign_manifest(decoded)


def _host_ids(manifest: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(host["id"]) for host in manifest["hosts"])  # type: ignore[index]


def _assignments(manifest: Mapping[str, object]) -> tuple[StageAssignment, ...]:
    host_ids = _host_ids(manifest)
    coordinator = str(manifest["coordinator"])
    assignments: list[StageAssignment] = []
    for spec in STAGE_DAG:
        targets = host_ids if spec.scope == "all_hosts" else (coordinator,)
        for host_id in targets:
            assignments.append(
                StageAssignment(
                    work_id=f"{spec.stage}:{host_id}",
                    stage=spec.stage,
                    host_id=host_id,
                    assignment_index=len(assignments),
                )
            )
    return tuple(assignments)


def _state_body(
    *,
    campaign_identity_sha256: str,
    completions: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": CAMPAIGN_STATE_SCHEMA,
        "campaign_identity_sha256": campaign_identity_sha256,
        "completions": completions,
    }


def _seal_state_body(body: dict[str, object]) -> dict[str, object]:
    return {**body, "identity_sha256": canonical_sha256(body)}


def initial_campaign_state(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Return a sealed empty state bound to ``manifest``."""

    normalized = validate_campaign_manifest(manifest)
    return _seal_state_body(
        _state_body(
            campaign_identity_sha256=str(normalized["identity_sha256"]),
            completions=[],
        )
    )


def _validate_completion(
    value: object,
    *,
    index: int,
    manifest: Mapping[str, object],
    assignment_by_work_id: Mapping[str, StageAssignment],
    host_by_id: Mapping[str, Mapping[str, object]],
    producer_identity: str,
) -> tuple[dict[str, object], StageAssignment]:
    where = f"campaign state completions[{index}]"
    raw = _exact_mapping(value, keys=_COMPLETION_KEYS, where=where)
    if raw["schema"] != STAGE_RECEIPT_SCHEMA:
        _fail(f"{where}.schema is unsupported")
    work_id = _string(raw["work_id"], where=f"{where}.work_id")
    assignment = assignment_by_work_id.get(work_id)
    if assignment is None:
        _fail(f"{where}.work_id is not in the fixed campaign DAG")
    stage = _string(raw["stage"], where=f"{where}.stage")
    host_id = _string(raw["host_id"], where=f"{where}.host_id")
    assignment_index = _integer(
        raw["assignment_index"],
        where=f"{where}.assignment_index",
        minimum=0,
    )
    if (
        stage != assignment.stage
        or host_id != assignment.host_id
        or assignment_index != assignment.assignment_index
    ):
        _fail(f"{where} does not match its deterministic assignment")
    campaign_identity = _sha256(
        raw["campaign_identity_sha256"],
        where=f"{where}.campaign_identity_sha256",
    )
    if campaign_identity != manifest["identity_sha256"]:
        _fail(f"{where} belongs to a different campaign manifest")
    host_identity = _sha256(
        raw["host_identity_sha256"], where=f"{where}.host_identity_sha256"
    )
    expected_host_identity = canonical_sha256(host_by_id[host_id])
    if host_identity != expected_host_identity:
        _fail(f"{where}.host_identity_sha256 differs from the manifest host")
    completion_producer_identity = _sha256(
        raw["producer_identity_sha256"],
        where=f"{where}.producer_identity_sha256",
    )
    if completion_producer_identity != producer_identity:
        _fail(f"{where}.producer_identity_sha256 differs from the manifest")
    receipt_sha256 = _sha256(
        raw["receipt_sha256"], where=f"{where}.receipt_sha256"
    )
    return (
        {
            "schema": STAGE_RECEIPT_SCHEMA,
            "work_id": work_id,
            "stage": stage,
            "host_id": host_id,
            "assignment_index": assignment_index,
            "campaign_identity_sha256": campaign_identity,
            "host_identity_sha256": host_identity,
            "producer_identity_sha256": completion_producer_identity,
            "receipt_sha256": receipt_sha256,
        },
        assignment,
    )


def _validate_state_against_normalized_manifest(
    value: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_STATE_KEYS, where="campaign state")
    if raw["schema"] != CAMPAIGN_STATE_SCHEMA:
        _fail("campaign state schema is unsupported")
    campaign_identity = _sha256(
        raw["campaign_identity_sha256"],
        where="campaign state campaign_identity_sha256",
    )
    if campaign_identity != manifest["identity_sha256"]:
        _fail("campaign state is bound to a different campaign manifest")
    raw_completions = raw["completions"]
    if type(raw_completions) is not list:
        _fail("campaign state completions must be a list")

    assignments = _assignments(manifest)
    assignment_by_work_id = {
        assignment.work_id: assignment for assignment in assignments
    }
    host_by_id = {
        str(host["id"]): host for host in manifest["hosts"]  # type: ignore[index]
    }
    producer_identity = canonical_sha256(manifest["producer"])
    completions: list[dict[str, object]] = []
    completion_assignments: list[StageAssignment] = []
    seen_work_ids: set[str] = set()
    for index, item in enumerate(raw_completions):
        completion, assignment = _validate_completion(
            item,
            index=index,
            manifest=manifest,
            assignment_by_work_id=assignment_by_work_id,
            host_by_id=host_by_id,
            producer_identity=producer_identity,
        )
        if assignment.work_id in seen_work_ids:
            _fail(f"campaign state contains duplicate work_id {assignment.work_id!r}")
        seen_work_ids.add(assignment.work_id)
        completions.append(completion)
        completion_assignments.append(assignment)

    indices = [item.assignment_index for item in completion_assignments]
    if indices != sorted(indices):
        _fail("campaign state completions are not in canonical assignment order")

    stage_work_ids: dict[str, frozenset[str]] = {
        spec.stage: frozenset(
            assignment.work_id
            for assignment in assignments
            if assignment.stage == spec.stage
        )
        for spec in STAGE_DAG
    }
    for assignment in completion_assignments:
        spec = _STAGE_BY_NAME[assignment.stage]
        for dependency in spec.dependencies:
            missing = stage_work_ids[dependency] - seen_work_ids
            if missing:
                _fail(
                    f"campaign state violates barrier before {assignment.stage}: "
                    f"missing {sorted(missing)}"
                )

    body = _state_body(
        campaign_identity_sha256=campaign_identity,
        completions=completions,
    )
    identity = _sha256(
        raw["identity_sha256"], where="campaign state identity_sha256"
    )
    if identity != canonical_sha256(body):
        _fail("campaign state identity_sha256 differs from its canonical body")
    return {**body, "identity_sha256": identity}


def validate_campaign_state(
    state: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate a sealed durable state against the exact campaign manifest."""

    normalized_manifest = validate_campaign_manifest(manifest)
    return _validate_state_against_normalized_manifest(state, normalized_manifest)


def _next_ready_from_validated(
    manifest: Mapping[str, object],
    state: Mapping[str, object],
) -> tuple[StageAssignment, ...]:
    completed = {
        str(item["work_id"]) for item in state["completions"]  # type: ignore[index]
    }
    assignments = _assignments(manifest)
    for spec in STAGE_DAG:
        stage_assignments = tuple(
            item for item in assignments if item.stage == spec.stage
        )
        remaining = tuple(
            item for item in stage_assignments if item.work_id not in completed
        )
        if remaining:
            return remaining
    return ()


def next_ready_assignments(
    manifest: Mapping[str, object],
    state: Mapping[str, object],
) -> tuple[StageAssignment, ...]:
    """Return the deterministic frontier, including all parallel host work."""

    normalized_manifest = validate_campaign_manifest(manifest)
    normalized_state = _validate_state_against_normalized_manifest(
        state, normalized_manifest
    )
    return _next_ready_from_validated(normalized_manifest, normalized_state)


def complete_assignment(
    manifest: Mapping[str, object],
    state: Mapping[str, object],
    assignment: StageAssignment,
    receipt_sha256: str,
) -> dict[str, object]:
    """Purely add one current-frontier completion and reseal the state."""

    if not isinstance(assignment, StageAssignment):
        _fail("assignment must be a StageAssignment")
    receipt_identity = _sha256(receipt_sha256, where="receipt_sha256")
    normalized_manifest = validate_campaign_manifest(manifest)
    normalized_state = _validate_state_against_normalized_manifest(
        state, normalized_manifest
    )
    ready = _next_ready_from_validated(normalized_manifest, normalized_state)
    if assignment not in ready:
        _fail("assignment is not in the current deterministic frontier")

    host_by_id = {
        str(host["id"]): host
        for host in normalized_manifest["hosts"]  # type: ignore[index]
    }
    completion = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "work_id": assignment.work_id,
        "stage": assignment.stage,
        "host_id": assignment.host_id,
        "assignment_index": assignment.assignment_index,
        "campaign_identity_sha256": normalized_manifest["identity_sha256"],
        "host_identity_sha256": canonical_sha256(
            host_by_id[assignment.host_id]
        ),
        "producer_identity_sha256": canonical_sha256(
            normalized_manifest["producer"]
        ),
        "receipt_sha256": receipt_identity,
    }
    completions = [
        dict(item) for item in normalized_state["completions"]  # type: ignore[arg-type]
    ]
    completions.append(completion)
    completions.sort(key=lambda item: int(item["assignment_index"]))
    return _seal_state_body(
        _state_body(
            campaign_identity_sha256=str(
                normalized_manifest["identity_sha256"]
            ),
            completions=completions,
        )
    )


__all__ = [
    "CAMPAIGN_MANIFEST_SCHEMA",
    "CAMPAIGN_STATE_SCHEMA",
    "CANONICAL_CONTAINER_PATHS",
    "ClusterCampaignContractError",
    "FIXED_ARTIFACT_TARGET",
    "FIXED_CAMPAIGN_POLICY",
    "GRIDBOOK_RUNTIME_CONTRACT_INPUT_SCHEMA",
    "LEGACY_CAMPAIGN_MANIFEST_SCHEMA",
    "LEGACY_FIXED_ARTIFACT_TARGET",
    "LEGACY_FIXED_CAMPAIGN_POLICY",
    "RTX4090_COMPUTE_CAPABILITY",
    "RTX4090_GPU_NAME",
    "STAGE_DAG",
    "STAGE_RECEIPT_SCHEMA",
    "StageAssignment",
    "StageSpec",
    "bind_gridbook_runtime_contract",
    "canonical_sha256",
    "complete_assignment",
    "initial_campaign_state",
    "gridbook_runtime_contract_sha256",
    "next_ready_assignments",
    "parse_campaign_manifest",
    "seal_campaign_manifest",
    "validate_campaign_manifest",
    "validate_campaign_state",
]
